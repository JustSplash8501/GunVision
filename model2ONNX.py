import os
import glob
from ultralytics import YOLO  
import onnx 
from onnxruntime.quantization import quantize_dynamic, QuantType 

# CONFIG
PT_MODEL_PATH = "runs/detect/pistol_model_v1_0/weights/best.pt"
MODEL_DIR = "models"

# Output paths
ONNX_BASELINE = os.path.join(MODEL_DIR, "pistol_detection_fp32.onnx")
ONNX_QUANTIZED = os.path.join(MODEL_DIR, "pistol_detection_int8.onnx")

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("YOLO Model Export & Quantization Pipeline")
print("=" * 60)

# --- Step 1: Load YOLO model ---
print(f"\n[1/3] Loading YOLO model from: {PT_MODEL_PATH}")
yolo_model = YOLO(PT_MODEL_PATH)

# --- Step 2: Export to ONNX (FP32) ---
print("\n[2/3] Exporting to ONNX format...")
export_result = yolo_model.export(
    format="onnx",
    opset=12,
    dynamic=False,
    simplify=True  # Simplified ONNX graph (note: requires onnxslim if available)
)

# Resolve exported ONNX path
onnx_candidates = []

if isinstance(export_result, (str, os.PathLike)):
    onnx_candidates = [str(export_result)]
elif isinstance(export_result, (list, tuple)):
    onnx_candidates = [str(p) for p in export_result if p]

# Fallback: search for .onnx files
if not onnx_candidates:
    onnx_candidates = glob.glob("*.onnx") + glob.glob(os.path.join(MODEL_DIR, "*.onnx"))

# Filter existing files and pick newest
onnx_candidates = [p for p in onnx_candidates if os.path.exists(p)]
onnx_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)

if not onnx_candidates:
    raise FileNotFoundError(
        "ONNX export did not produce a .onnx file. "
        "Check the Ultralytics export logs for errors."
    )

EXPORT_PATH = onnx_candidates[0]
print(f"   Exported to: {EXPORT_PATH}")

# Save as baseline model
onnx_model = onnx.load(EXPORT_PATH)
onnx.save(onnx_model, ONNX_BASELINE)
print(f"   Baseline saved: {ONNX_BASELINE}")

# Get baseline size
baseline_size = os.path.getsize(ONNX_BASELINE) / (1024 * 1024)  # MB
print(f"   Baseline size: {baseline_size:.2f} MB")

# --- Step 3: Quantize to INT8 ---
print("\n[3/3] Quantizing model to INT8...")
quantize_dynamic(
    model_input=ONNX_BASELINE,
    model_output=ONNX_QUANTIZED,
    weight_type=QuantType.QUInt8
)
print(f"   Quantized model saved: {ONNX_QUANTIZED}")

# Get quantized size and compression ratio
quantized_size = os.path.getsize(ONNX_QUANTIZED) / (1024 * 1024)  # MB
compression_ratio = (1 - quantized_size / baseline_size) * 100
print(f"   Quantized size: {quantized_size:.2f} MB")
print(f"   Compression: {compression_ratio:.1f}% reduction")

# Summary
print("\n" + "=" * 60)
print("Export Complete! Models saved:")
print("=" * 60)
print(f"FP32 Baseline:  {ONNX_BASELINE} ({baseline_size:.2f} MB)")
print(f"INT8 Quantized: {ONNX_QUANTIZED} ({quantized_size:.2f} MB)")

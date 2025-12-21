from ultralytics import YOLO # type: ignore
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.transformers.optimizer import optimize_model

model = YOLO('path/to/your/best.pt')

model.export(format='onnx', opset=12)

model = onnx.load("pistol_detection.onnx")

# Optimize
optimized_model = optimize_model("pistol_detection.onnx")
optimized_model.save_model_to_file("pistol_detection_optimized.onnx")

# Quantize for even faster inference (INT8)
quantize_dynamic(
    "pistol_detection_optimized.onnx",
    "pistol_detection_quantized.onnx",
    weight_type=QuantType.QUInt8
)

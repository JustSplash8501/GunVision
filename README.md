# Pistol Detection Model

A YOLOv11-based computer vision system for firearm detection in images and video streams, optimized for edge deployment through INT8 quantization.

## Dataset

**Source:** [Pistols Dataset - Roboflow](https://public.roboflow.ai/object-detection/pistols)  
**Original Provider:** [University of Granada - Weapons Detection Research](https://sci2s.ugr.es/weapons-detection#RP)  
**License:** Public Domain  
**Version:** v1 (resize-416x416)

### Dataset Statistics
- **Total Images:** 2,973
- **Total Annotations:** 3,448 bounding boxes
- **Classes:** 1 (pistol)
- **Annotation Format:** YOLOv10 YAML
- **Image Resolution:** 416x416 pixels (stretched)

The dataset includes diverse imagery: handheld pistols, illustrations, studio photographs, and real-world scenarios. All images have been preprocessed with auto-orientation correction and resized to 416x416 for training consistency.

![Dataset Example](https://i.imgur.com/vX7WoRO.png)

## Model Architecture

**Base Model:** YOLOv11 (Ultralytics)  
**Training Resolution:** 416x416 pixels  
**Export Format:** ONNX  
**Quantization:** INT8 dynamic quantization  
**Inference Engine:** ONNX Runtime (CPU)

Further training specifics can be found in `notebooks/training.ipynb`.

### Optimization Pipeline

The optimization process consisted of three stages:

1. **Baseline Export:** Convert PyTorch model to ONNX FP32 format
2. **Graph Optimization:** Apply ONNX Runtime graph optimizations
3. **Quantization:** Dynamic INT8 quantization for weight compression

The quantized model achieves a 75% size reduction (27.8 MB → 7.1 MB) while maintaining competitive accuracy and improving CPU inference throughput by approximately 20%.

For detailed performance metrics, optimization analysis, and model comparisons, see `notebooks/modelCompare.qmd`.

A comparison of the original, baseline ONNX model, an optimized FP32 model, and an optimized and quantized INT8 model can be found in `notebooks/modelComparison.qmd` using the three models in `models`.

### Performance

- **mAP@50:** 0.895 (measured on the validation set)
- **Inference throughput:** ~15 FPS on CPU (ONNX Runtime)
- **Train/Val split:** 80/20
- **Model variant used:** `s` (small) variation of YOLOv11
- **Throughput comparison note:** the reported ~20% CPU throughput improvement compares inference between the optimized/quantized model and the baseline model (model-to-model comparison), not different hardware configurations.

### Limitations & Intended Use

- **Dataset size and diversity:** The training dataset contains 2,973 images and 3,448 annotations. This is a modest dataset and may not capture the full variety of real-world appearances or imaging conditions (different cameras, lighting, occlusion patterns, wear, unusual viewpoints).
- **Quantization effects:** INT8 quantization reduces model size and typically improves throughput, but may slightly alter per-class accuracy and robustness on edge cases—validate on your own data before deployment.
- **Hardware variation:** CPU throughput and latency depend strongly on CPU microarchitecture, available SIMD instruction sets (AVX2), and the ONNX Runtime build; measured numbers are a reference, not a guarantee.
- **Domain shift:** Performance will degrade if the deployment domain differs from the training/validation data (different image resolutions, camera angles, occlusion, or new object types). Collect and fine-tune on in-domain data when possible.

What this model is NOT for:

- **Not for sole decision-making in law enforcement.** Do not use this model as the only source of evidence for arrests, detentions, searches, or other policing decisions.
- **Not for automated enforcement.** It must not be used for automated ticketing, automated restriction enforcement, or any automated punitive actions.
- **Not for unsupervised real-world deployment.** Do not deploy in production-critical or safety-critical systems without human oversight, human-review queues, and extensive domain-specific validation.
- **Not a substitute for full investigative processes.** Outputs are model predictions and require human interpretation, verification, and contextual judgment.

If you plan to use this model in operational settings, you should conduct end-to-end evaluation on representative in-domain data, implement a human-in-the-loop review process, and validate legal/ethical compliance for your jurisdiction and use case.

## Installation

### Prerequisites
- Python 3.11+
- CPU with AVX2 support (for optimal ONNX Runtime performance)

### Setup
```bash
git clone https://github.com/JustSplash8501/GunVision.git
cd GunVision

# Using uv (recommended)
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Web Application

Launch the Gradio interface for interactive inference:
```bash
python app/app.py
```

The application provides three inference modes:
- **Image Upload:** Single image detection with detailed statistics
- **Webcam Stream:** Real-time detection at ~10 FPS
- **Video Processing:** Frame-by-frame analysis with aggregated metrics

Access the interface at `http://localhost:7860`

Repository tree snapshot: see `CODE_TREE.md` at the repository root for a quick directory overview.

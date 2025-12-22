import gradio as gr
import cv2
import numpy as np
from PIL import Image
import tempfile
import onnxruntime as ort

# Load ONNX model
MODEL_PATH = "models/pistol_detection_int8.onnx"
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

# Get model input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
model_height, model_width = input_shape[2], input_shape[3]

# YOLO class names (adjust if your model has different classes)
CLASS_NAMES = ['pistol']


def preprocess_image(image, target_size=(640, 640)):
    """Preprocess image for YOLO model"""
    # Resize image
    resized = cv2.resize(image, target_size)
    
    # Convert to RGB if needed
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    elif resized.shape[2] == 4:
        resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)
    
    # Normalize to [0, 1] and transpose to CHW format
    input_tensor = resized.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    return input_tensor


def postprocess_detections(outputs, original_shape, conf_threshold=0.5):
    """
    Process YOLO outputs and return detections
    
    Args:
        outputs: Model output array (already post-processed with NMS)
        original_shape: (height, width) of original image
        conf_threshold: Confidence threshold for filtering
    
    Returns:
        List of detections: [x1, y1, x2, y2, confidence, class_id]
    """
    # Output shape: [1, 300, 6] where 6 = [x1, y1, x2, y2, confidence, class_id]
    predictions = outputs[0]
    
    # Remove batch dimension
    if len(predictions.shape) == 3:
        predictions = predictions[0]
    
    detections = []
    orig_h, orig_w = original_shape
    
    # Scale factors from model input size to original image size
    scale_x = orig_w / model_width
    scale_y = orig_h / model_height
    
    for pred in predictions:
        # Extract values: [x1, y1, x2, y2, confidence, class_id]
        x1, y1, x2, y2, confidence, class_id = pred
        
        # Filter by confidence threshold
        if confidence >= conf_threshold:
            # Scale coordinates from 640x640 to original image size
            x1_scaled = x1 * scale_x
            y1_scaled = y1 * scale_y
            x2_scaled = x2 * scale_x
            y2_scaled = y2 * scale_y
            
            # Clip to image boundaries
            x1_scaled = max(0, min(x1_scaled, orig_w))
            y1_scaled = max(0, min(y1_scaled, orig_h))
            x2_scaled = max(0, min(x2_scaled, orig_w))
            y2_scaled = max(0, min(y2_scaled, orig_h))
            
            detections.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled, confidence, int(class_id)])
    
    return detections


def draw_detections(image, detections, font_scale=0.6, thickness=2, box_thickness=2):
    """
    Draw bounding boxes on image
    
    Args:
        image: Input image
        detections: List of detections
        font_scale: Font size (default 0.6, increase for larger text)
        thickness: Text thickness (default 2)
        box_thickness: Bounding box line thickness (default 2)
    """
    annotated = image.copy()
    
    for det in detections:
        x1, y1, x2, y2, conf, class_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)
        
        # Draw label
        label = f"{CLASS_NAMES[int(class_id)]}: {conf:.2%}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Draw label background
        cv2.rectangle(
            annotated,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            (0, 255, 0),
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness
        )
    
    return annotated


def detect_pistol_image(image, confidence_threshold):
    """
    Detect pistols in a single image
    
    Args:
        image: PIL Image or numpy array
        confidence_threshold: float between 0 and 1
    
    Returns:
        annotated_image: Image with detections drawn
        detection_info: String with detection details
    """
    if image is None:
        return None, "No image provided"
    
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure RGB format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    original_shape = image.shape[:2]
    
    # Preprocess
    input_tensor = preprocess_image(image, (model_width, model_height))
    
    # Run inference
    outputs = session.run(None, {input_name: input_tensor})
    
    # Debug: Print output shape
    output_shape = outputs[0].shape
    
    # Postprocess
    detections = postprocess_detections(outputs, original_shape, confidence_threshold)
    
    # Draw detections
    annotated_image = draw_detections(image, detections)
    
    # Create detection info with debug information
    detection_info = f"Model: {MODEL_PATH}\n"
    detection_info += f"Input shape: {input_tensor.shape}\n"
    detection_info += f"Output shape: {output_shape}\n"
    detection_info += f"Confidence threshold: {confidence_threshold}\n"
    detection_info += f"Total detections: {len(detections)}\n\n"
    
    if len(detections) > 0:
        detection_info += "Detections:\n"
        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2, conf, class_id = det
            detection_info += f"\n{i}. {CLASS_NAMES[int(class_id)].capitalize()}\n"
            detection_info += f"   Confidence: {conf:.2%}\n"
            detection_info += f"   Bounding box: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})\n"
            detection_info += f"   Size: {int(x2-x1)}x{int(y2-y1)} pixels\n"
    else:
        detection_info += "No pistols detected above confidence threshold\n"
        detection_info += f"\nTry lowering the confidence threshold or check if the model is detecting correctly."
    
    return annotated_image, detection_info


def detect_pistol_video(video_path, confidence_threshold, progress=gr.Progress()):
    """
    Detect pistols in video and return processed video
    
    Args:
        video_path: Path to input video
        confidence_threshold: float between 0 and 1
        progress: Gradio progress tracker
    
    Returns:
        output_video_path: Path to processed video
        detection_summary: String with overall detection statistics
    """
    if video_path is None:
        return None, "No video uploaded"
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    frames_with_detections = 0
    max_detections_per_frame = 0
    
    progress(0, desc="Processing video...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_shape = frame_rgb.shape[:2]
        
        # Preprocess
        input_tensor = preprocess_image(frame_rgb, (model_width, model_height))
        
        # Run inference
        outputs = session.run(None, {input_name: input_tensor})
        
        # Postprocess
        detections = postprocess_detections(outputs, original_shape, confidence_threshold)
        num_detections = len(detections)
        
        # Draw detections
        annotated_frame = draw_detections(frame_rgb, detections)
        
        # Convert back to BGR for video writer
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        out.write(annotated_frame_bgr)
        
        # Update statistics
        if num_detections > 0:
            frames_with_detections += 1
            total_detections += num_detections
            max_detections_per_frame = max(max_detections_per_frame, num_detections)
        
        frame_count += 1
        progress(frame_count / total_frames, desc=f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    # Create detection summary
    detection_summary = "Video Processing Complete\n\n"
    detection_summary += f"Model: {MODEL_PATH}\n"
    detection_summary += f"Total frames: {total_frames}\n"
    detection_summary += f"Frames with detections: {frames_with_detections} ({frames_with_detections/total_frames*100:.1f}%)\n"
    detection_summary += f"Total detections: {total_detections}\n"
    detection_summary += f"Max detections per frame: {max_detections_per_frame}\n"
    detection_summary += f"Confidence threshold: {confidence_threshold}\n"
    
    if frames_with_detections > 0:
        avg_detections = total_detections / frames_with_detections
        detection_summary += f"Average detections per frame (when detected): {avg_detections:.2f}\n"
    
    return output_path, detection_summary


# Load custom CSS
def load_css():
    """Load custom CSS from external file"""
    try:
        with open('app/styles.css', 'r') as f:
            return f.read()
    except FileNotFoundError:
        print("Warning: styles.css not found, using inline styles")
        return ""

# Create Gradio interface with tabs
with gr.Blocks(title="Pistol Detection System") as demo:
    
    # Header with GitHub link
    gr.Markdown("# 🎯 Pistol Detection System")
    gr.HTML("""
        <div class="subtitle">
            Advanced firearm detection powered by INT8-quantized YOLOv8 neural network.
            Real-time inference optimized for edge deployment.
        </div>
        <div style="text-align: center; margin-bottom: 20px;">
            <a href="https://github.com/yourusername/pistol-detection" target="_blank" class="github-link">
                <svg height="20" width="20" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                </svg>
                View on GitHub
            </a>
        </div>
    """)
    
    # Statistics row
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            gr.HTML("""
            <div class="stat-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <h3>⚡ Model Stats</h3>
                <p class="stat-value">7.1 MB</p>
                <p class="stat-label">Compressed Size</p>
            </div>
            """)
        with gr.Column(scale=1, min_width=200):
            gr.HTML("""
            <div class="stat-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <h3>🚀 Performance</h3>
                <p class="stat-value">~30 FPS</p>
                <p class="stat-label">CPU Inference</p>
            </div>
            """)
        with gr.Column(scale=1, min_width=200):
            gr.HTML("""
            <div class="stat-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <h3>🎯 Accuracy</h3>
                <p class="stat-value">89.5%</p>
                <p class="stat-label">mAP@50</p>
            </div>
            """)
    
    gr.Markdown("<br>")
    
    with gr.Tabs():
        # Image Upload Tab
        with gr.Tab("📸 Image Upload"):
            gr.HTML("""
            <div class="info-box" style="background-color: #f8f9fa; border-left: 4px solid #667eea;">
                <strong>💡 Tip:</strong> Upload clear images with visible firearms for best results. 
                Supported formats: JPG, PNG, WebP
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    image_input = gr.Image(
                        sources=["upload"],
                        type="numpy",
                        label="Upload Image"
                    )
                    confidence_slider_image = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="🎚️ Confidence Threshold",
                        info="Lower values detect more objects but may include false positives"
                    )
                    detect_image_btn = gr.Button(
                        "🔍 Detect Pistols",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1, min_width=300):
                    image_output = gr.Image(label="Detection Results")
                    detection_text = gr.Textbox(
                        label="📊 Detection Details",
                        lines=12,
                        max_lines=20
                    )
            
            detect_image_btn.click(
                fn=detect_pistol_image,
                inputs=[image_input, confidence_slider_image],
                outputs=[image_output, detection_text]
            )
        
        # Webcam Tab (Real-time)
        with gr.Tab("📹 Webcam (Real-time)"):
            gr.HTML("""
            <div class="info-box" style="background-color: #fff3cd; border-left: 4px solid #ffc107;">
                <strong>⚠️ Note:</strong> Real-time streaming processes frames at ~10 FPS. 
                Allow camera permissions when prompted.
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        label="Webcam Feed",
                        streaming=True
                    )
                    confidence_slider_webcam = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="🎚️ Confidence Threshold",
                        info="Adjust sensitivity for real-time detection"
                    )
                
                with gr.Column(scale=1, min_width=300):
                    webcam_output = gr.Image(label="Live Detection Results")
                    webcam_text = gr.Textbox(
                        label="📊 Detection Details",
                        lines=12,
                        max_lines=20
                    )
            
            webcam_input.stream(
                fn=detect_pistol_image,
                inputs=[webcam_input, confidence_slider_webcam],
                outputs=[webcam_output, webcam_text],
                stream_every=0.1
            )
        
        # Video Tab
        with gr.Tab("🎬 Video Processing"):
            gr.HTML("""
            <div class="info-box" style="background-color: #d1ecf1; border-left: 4px solid #17a2b8;">
                <strong>ℹ️ Info:</strong> Upload MP4 videos for frame-by-frame analysis. 
                Processing time depends on video length and resolution.
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    video_input = gr.Video(
                        label="Upload Video",
                        sources=["upload"]
                    )
                    confidence_slider_video = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="🎚️ Confidence Threshold",
                        info="Filter detections by confidence level"
                    )
                    detect_video_btn = gr.Button(
                        "▶️ Process Video",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1, min_width=300):
                    video_output = gr.Video(label="Processed Video")
                    video_summary = gr.Textbox(
                        label="📊 Detection Summary",
                        lines=12
                    )
            
            detect_video_btn.click(
                fn=detect_pistol_video,
                inputs=[video_input, confidence_slider_video],
                outputs=[video_output, video_summary]
            )
    
    # Instructions Section
    gr.Markdown("---")
    gr.Markdown("## 📚 How It Works")
    
    with gr.Row():
        with gr.Column(min_width=250):
            gr.Markdown("""
            ### 1️⃣ Upload Content
            Choose from three input methods:
            - Static images (JPG, PNG)
            - Live webcam stream
            - Pre-recorded videos (MP4)
            """)
        with gr.Column(min_width=250):
            gr.Markdown("""
            ### 2️⃣ Adjust Settings
            Fine-tune detection sensitivity:
            - Higher threshold: Fewer, more confident detections
            - Lower threshold: More detections, potential false positives
            """)
        with gr.Column(min_width=250):
            gr.Markdown("""
            ### 3️⃣ Analyze Results
            View comprehensive detection data:
            - Bounding box coordinates
            - Confidence scores
            - Detection statistics
            """)
    
    # Model Information
    gr.Markdown("---")
    gr.Markdown("## 🔧 Technical Specifications")
    
    with gr.Accordion("Model Architecture", open=False):
        gr.Markdown("""
        **Base Model:** YOLOv8 (You Only Look Once version 8)  
        **Optimization:** INT8 Dynamic Quantization  
        **Input Resolution:** 640 × 640 pixels  
        **Output Format:** Bounding boxes with confidence scores  
        **Inference Engine:** ONNX Runtime (CPU)  
        **Training Dataset:** 446 images, 512 pistol instances  
        **Model Size:** 7.1 MB (75% reduction from FP32)  
        
        **Performance Metrics:**
        - Precision: 85.4%
        - Recall: 82.6%
        - mAP@50: 89.5%
        - mAP@50-95: 64.6%
        """)
    
    # Footer
    gr.HTML("""
    <footer>
        <p style="color: #666; margin: 0; font-size: clamp(0.9em, 1.5vw, 1em);">
            Built with Gradio • Powered by ONNX Runtime • Optimized for Edge Deployment
        </p>
        <p style="color: #999; font-size: clamp(0.8em, 1.3vw, 0.9em); margin-top: 10px;">
            Model trained on custom pistol detection dataset • For educational and security purposes
        </p>
        <div style="margin-top: 15px;">
            <a href="https://github.com/yourusername/pistol-detection" target="_blank" class="github-link">
                <svg height="20" width="20" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                </svg>
                Contribute on GitHub
            </a>
        </div>
    </footer>
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=False,
        theme=gr.themes.Soft(
            primary_hue="red",
            secondary_hue="gray",
            neutral_hue="slate",
        ),
        css=load_css()
    )

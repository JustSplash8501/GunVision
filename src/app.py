import gradio as gr
import cv2
import numpy as np
from PIL import Image
import tempfile

# TODO: Replace with your actual model loading
# model = YOLO('path/to/your/model.pt')

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
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # TODO: Replace with your actual model inference
    # Example for YOLOv8:
    # results = model(image, conf=confidence_threshold)
    # annotated_image = results[0].plot()
    
    # Example for custom model:
    # predictions = model.predict(image)
    # annotated_image = draw_predictions(image, predictions, confidence_threshold)
    
    # PLACEHOLDER: Remove this and use your model
    annotated_image = image.copy()
    detection_info = "Model not loaded. Replace with your trained model.\n\n"
    detection_info += f"Confidence threshold: {confidence_threshold}\n"
    detection_info += "Instructions:\n"
    detection_info += "1. Load your trained model at the top of the script\n"
    detection_info += "2. Run inference in this function\n"
    detection_info += "3. Draw bounding boxes/segmentation masks\n"
    detection_info += "4. Return annotated image and detection details"
    
    # Example of what detection_info should look like:
    # detection_info = f"Detections: {len(detections)}\n\n"
    # for i, det in enumerate(detections):
    #     detection_info += f"Detection {i+1}:\n"
    #     detection_info += f"  Confidence: {det.confidence:.2%}\n"
    #     detection_info += f"  Bbox: {det.bbox}\n\n"
    
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
    
    progress(0, desc="Processing video...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # TODO: Replace with your actual model inference
        # Example for YOLOv8:
        # results = model(frame_rgb, conf=confidence_threshold)
        # annotated_frame = results[0].plot()
        # num_detections = len(results[0].boxes)
        
        # PLACEHOLDER: Remove this and use your model
        annotated_frame = frame_rgb
        num_detections = 0
        
        # Convert back to BGR for video writer
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        out.write(annotated_frame_bgr)
        
        # Update statistics
        if num_detections > 0:
            frames_with_detections += 1
            total_detections += num_detections
        
        frame_count += 1
        progress(frame_count / total_frames, desc=f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    # Create detection summary
    detection_summary = "Video Processing Complete\n\n"
    detection_summary += f"Total Frames: {total_frames}\n"
    detection_summary += f"Frames with Detections: {frames_with_detections}\n"
    detection_summary += f"Total Detections: {total_detections}\n"
    detection_summary += f"Confidence Threshold: {confidence_threshold}\n"
    
    if frames_with_detections > 0:
        avg_detections = total_detections / frames_with_detections
        detection_summary += f"Avg Detections per Frame (with detections): {avg_detections:.2f}\n"
    
    return output_path, detection_summary


# Create Gradio interface with tabs
with gr.Blocks(title="Pistol Detection & Segmentation") as demo:
    gr.Markdown("# 🔫 Pistol Detection & Segmentation")
    gr.Markdown("Upload an image or video, use your webcam, or try example images to detect and segment pistols.")
    
    with gr.Tabs():
        # Image/Webcam Tab
        with gr.Tab("Image/Webcam"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        sources=["webcam", "upload"],
                        type="numpy",
                        label="Input Image"
                    )
                    confidence_slider_image = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="Confidence Threshold"
                    )
                    detect_image_btn = gr.Button("Detect Pistols", variant="primary")
                
                with gr.Column():
                    image_output = gr.Image(label="Detected Results")
                    detection_text = gr.Textbox(
                        label="Detection Details",
                        lines=10,
                        max_lines=15
                    )
            
            detect_image_btn.click(
                fn=detect_pistol_image,
                inputs=[image_input, confidence_slider_image],
                outputs=[image_output, detection_text]
            )
        
        # Video Tab
        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(
                        label="Upload Video (MP4)",
                        sources=["upload"]
                    )
                    confidence_slider_video = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="Confidence Threshold"
                    )
                    detect_video_btn = gr.Button("Process Video", variant="primary")
                
                with gr.Column():
                    video_output = gr.Video(label="Processed Video")
                    video_summary = gr.Textbox(
                        label="Detection Summary",
                        lines=10
                    )
            
            detect_video_btn.click(
                fn=detect_pistol_video,
                inputs=[video_input, confidence_slider_video],
                outputs=[video_output, video_summary]
            )
    
    # Optional: Add example images
    gr.Markdown("## 📝 Instructions")
    gr.Markdown("""
    1. **Image/Webcam Tab**: Upload an image or use your webcam to detect pistols in real-time
    2. **Video Tab**: Upload an MP4 video file to process all frames
    3. Adjust the confidence threshold to filter detections
    4. View confidence scores and detection counts in the details panel
    
    **Note**: Replace the placeholder model loading code with your trained model for actual detections.
    """)

# Launch the app
if __name__ == "__main__":
    # share=True creates a public link (optional)
    demo.launch(share=False)

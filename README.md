# Pistol Detection Model

A computer vision model for detecting pistols in images and video using deep learning-based object detection.

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

## Applications

This model can be deployed for various security and monitoring applications:

- **Security Camera Monitoring:** Real-time detection of firearms in restricted areas
- **Threat Assessment:** Automated alerts for security personnel
- **Video Analytics:** Post-incident analysis of surveillance footage
- **Access Control:** Enhanced security screening systems

## Preprocessing Pipeline

Each image underwent the following preprocessing steps:
- EXIF-based auto-orientation with metadata stripping
- Resize to 416x416 pixels (stretch mode)
- No augmentation applied in base dataset

## Model Deployment

This repository includes a Gradio web application for easy model deployment and testing. See the application code for inference on images, webcam feeds, and video files.

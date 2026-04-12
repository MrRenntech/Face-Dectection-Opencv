# 👁️ Face Detection OpenCV

<div align="center">
  ![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv)
  ![Python](https://img.shields.io/badge/Python-Script-blue?style=for-the-badge&logo=python)
</div>

Real-time face detection app using OpenCV's Haar Cascades.

---

## 🏗️ Pipeline Architecture

```mermaid
graph LR
    Cam[Live WebCam stream] --> CV[OpenCV VideoCapture]
    CV --> Gray[Grayscale Conversion]
    Gray --> Haar[Haar Cascade Classifier]
    Haar --> Rect[Draw Green Bounding Box]
    Rect --> UI[cv2.imshow]
```

## 🚀 Quick Start
```bash
install_requirements.bat
launch_app.bat
```

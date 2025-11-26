# 🚀 YOLO Multi-Task Web App

<div align="center">
<img src="./preview.png" width="80%" alt="YOLO Multi-Task Preview">

<br>

[![ONNX Runtime Web](https://img.shields.io/badge/ONNX%20Runtime-Web-blue)](https://onnxruntime.ai/)
[![YOLO](https://img.shields.io/badge/YOLO-v11%2Fv12-green)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

## 📖 Introduction

This web application, built on **ONNX Runtime Web**, brings the power of YOLO directly to your browser. It supports full client-side inference for Object Detection, Pose Estimation, and Instance Segmentation without sending data to a server.

## ⚠️ WebGPU Prerequisites (Important)

To achieve the best performance using **WebGPU**, please ensure the following:

1.  **Browser**: Use a Chromium-based browser (Chrome, Edge, Brave).
2.  **Enable Flags**:
    - Type `chrome://flags` (or `edge://flags`) in your address bar.
    - Search for **"Unsafe WebGPU Support"** and set it to **Enabled**.
    - **(Linux / Android users)**: Search for **"Vulkan"** (`#enable-vulkan`) and set it to **Enabled**.
    - Relaunch your browser.

> 💡 **Note**: If WebGPU is not available, the app will automatically fall back to WASM (CPU), which is slower but universally compatible.

## ✨ Features

- 🔍 **Object Detection** - Precisely identify and locate various objects.
- 👤 **Pose Estimation** - Real-time human keypoint tracking.
- 🖼️ **Instance Segmentation** - Pixel-level object masking and identification.
- ⚡ **High Performance** - Powered by WebGPU acceleration.

## 📹 Input Support

| Input Type         |  Format  | Use Case                                  |
| :----------------- | :------: | :---------------------------------------- |
| 📷 **Image**       | JPG, PNG | Single image analysis & batch processing. |
| 📹 **Video**       |   MP4    | Offline video analysis & content review.  |
| 📺 **Live Camera** |  Stream  | Real-time monitoring & interactive demos. |

## 📊 Supported Models

| Model        | Input Size | Params | mAP<sup>val<br>50-95 | Speed (ms)<br><sup>T4 TensorRT10 | Recommended For       |
| :----------- | :--------- | :----- | :------------------- | :------------------------------- | :-------------------- |
| **YOLO11-N** | 640        | 2.6M   | 39.5                 | 1.5                              | 📱 Mobile / Real-time |
| **YOLO11-S** | 640        | 9.4M   | 47.0                 | 2.5                              | 🖥️ High Accuracy      |
| **YOLO12-N** | 640        | 2.6M   | 40.6                 | 1.64                             | 📱 Mobile / Real-time |
| **YOLO12-S** | 640        | 9.3M   | 48.0                 | 2.61                             | 🖥️ High Accuracy      |

_Models are licensed under [AGPL-3.0](./public/models/LICENSE.txt) via [Ultralytics](https://github.com/ultralytics/ultralytics)._

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/nomi30701/yolo-multi-task-onnxruntime-web.git
   cd yolo-multi-task-onnxruntime-web
   ```

2. **Install dependencies**

   ```bash
   yarn install
   ```

3. **Run Development Server**

   ```bash
   yarn dev
   ```

4. **Build for Production**
   ```bash
   yarn build
   ```

## 🔧 Custom Models Guide

You can run your own YOLO models in this app.

### Step 1: Export to ONNX

Use Ultralytics to export your model. **Crucial:** Use `opset=12` for WebGPU compatibility.

```python
from ultralytics import YOLO

model = YOLO("path/to/your/model.pt")
# Export with opset=12 and dynamic shape
model.export(format="onnx", opset=12, dynamic=True)
```

### Step 2: Load your Model

You have two ways to load your model:

#### Option A: Quick Test (UI Upload)

Simply click the **"Add model"** button in the web interface to upload your `.onnx` file temporarily.

#### Option B: Permanent Integration (Code)

1. Copy your `.onnx` file to `./public/models/`.
2. Edit `App.jsx` to add your model to the list:

```jsx
<select name="model-selector">
  <option value="yolo11n">yolo11n-2.6M</option>
  <option value="your-custom-model">Your Custom Model</option>
</select>
```

### Step 3: Update Classes

If your model uses custom classes (not COCO), you need to update the class definitions:

- **UI Method**: Click **"Add Classes.json"** to upload a JSON file mapping class IDs to names.
- **Code Method**: Update `src/utils/yolo_classes.json`.

```json
{
  "class": {
    "0": "person",
    "1": "bicycle"
  }
}
```

## ⚙️ Configuration: Image Processing

You can control how images are pre-processed via the `imgsz_type` setting:

- **Dynamic (Default)**:

  - Uses the original image aspect ratio.
  - **Pros**: Best accuracy.
  - **Cons**: Slower on large images; inference time varies.
  - _Requires model exported with `dynamic=True`._

- **Zero Pad (Square)**:
  - Pads image to square and resizes to 640x640.
  - **Pros**: Consistent, faster speed suitable for real-time video.
  - **Cons**: Slight accuracy drop on small objects due to scaling.

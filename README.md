# 🚀 YOLO Multi-Task Web App

<div align="center">
<img src="https://github.com/nomi30701/yolo-multi-task-onnxruntime-web/blob/main/preview.png" width="80%" alt="YOLO Multi-Task Preview">
</div>

## ✨ Features

This web application built on ONNX Runtime Web implements YOLO's multi-task inference capabilities:

- 🔍 **Object Detection** - Precisely identify and locate various objects
- 👤 **Pose Estimation** - Track human keypoints and poses
- 🖼️ **Instance Segmentation** - Pixel-level object area identification

## 💻 Technical Support

- ⚡ **WebGPU Acceleration** - Leverage the latest Web graphics API for enhanced performance
- 🧠 **WASM (CPU)** - Provide compatibility on devices that don't support WebGPU
  
  
## 📊 Available Models
| Model                                                  | Input Size | Param. | Best For |
| :----------------------------------------------------- | :--------: | :----: | :------: |
| [YOLO11-N](https://github.com/ultralytics/ultralytics) |    640     |  2.6M  | 📱 Mobile devices & real-time applications |
| [YOLO11-S](https://github.com/ultralytics/ultralytics) |    640     |  9.4M  | 🖥️ Higher accuracy requirements |

## 🛠️ Installation Guide
```bash
# Clone repo
git clone https://github.com/nomi30701/yolo-multi-class-onnxruntime-web.git

# Navigate to the project directory
cd yolo-multi-class-onnxruntime-web

# Install dependencies
yarn install 
```
## 🚀 Running the Project
```bash
# Start development server
yarn dev

# Build the project
yarn build
```

## 🔧 Using Custom YOLO Models
### Step 1: Convert model to ONNX format
Read more on [Ultralytics](https://docs.ultralytics.com/).
  ```Python
  from ultralytics import YOLO

  # Load a model
  model = YOLO("yolo11n-pose.pt")

  # Export the model
  # Important: Use opset=12 to ensure WebGPU compatibility
  model.export(format="onnx", opset=12, dynamic=True)  
  ```

### Step 2: Add to the project
Choose one of the following methods:
  - 📁 Copy model to the `./public/models/` directory
  - 🔄 Upload directly through the **Add model** button in the web interface

### Step 3: Register model in App.jsx
  ```HTML
  <select name="model-selector" ref={modelRef} onChange={onModelChange}>
  {/* Add your model here */}
  <option value="YOUR_MODEL_NAME">Your Custom Model</option>
  <option value="yolo11n">yolo11n-2.6M</option>
  <option value="yolo11s">yolo11s-9.4M</option>
  </select>
  ```

### Step 4: Refresh and select your new model 🎉

## 💡 Advanced Configuration Tips
> 📏 **Dynamic Input Size**
> Dynamic input size support is enabled by default. For fixed size, modify `/utils/inference_pipeline.js`:
> 1. Uncomment this code:
> ```Javascript
> const [src_mat_preProcessed, xRatio, yRatio] = await preProcess(
>   src_mat,
>   sessionsConfig.input_shape[2],
>   sessionsConfig.input_shape[3]
> );
> ```
> 
> 2. Remove the dynamic sizing code:
> ```Javascript
> const [src_mat_preProcessed, div_width, div_height] = preProcess_dynamic(src_mat);
> const xRatio = src_mat.cols / div_width;
> const yRatio = src_mat.rows / div_height;
> ```
>
> 3. Change Tensor size
> ```Javascript
> const input_tensor = new ort.Tensor("float32", src_mat_preProcessed.data32F, [
>   1,
>   3,
>   config.input_shape[2],
>   config.input_shape[3],
> ]);
>


> 🚀 WebGPU Support
>
> Ensure you set `opset=12` when exporting ONNX models, as this is required for WebGPU compatibility.


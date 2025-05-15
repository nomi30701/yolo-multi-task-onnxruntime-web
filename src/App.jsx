import "./assets/App.css";
import cv from "@techstark/opencv-js";
import { useEffect, useRef, useState, useCallback } from "react";
import { model_loader } from "./utils/model_loader";
import { inference_pipeline } from "./utils/inference_pipeline";
import { draw_bounding_boxes } from "./utils/draw_bounding_boxes";
import classes from "./utils/yolo_classes.json";

const MODEL_CONFIG = {
  input_shape: [1, 3, 640, 640],
  iou_threshold: 0.35,
  score_threshold: 0.45,
  backend: "wasm",
  model: "yolo11n",
  model_path: "",
  task: "detect",
  imgsz_type: "dynamic",
};

// set Components
function SettingsPanel({
  backendSelectorRef,
  modelSelectorRef,
  taskSelectorRef,
  cameraSelectorRef,
  imgszTypeSelectorRef,
  modelConfigRef,
  cameras,
  customModels,
  loadModel,
  activeFeature,
}) {
  return (
    <div
      id="setting-container"
      className="container text-lg flex flex-col md:flex-row md:justify-evenly gap-2 md:gap-6"
    >
      <div id="selector-container">
        <label>Backend:</label>
        <select
          name="device-selector"
          ref={backendSelectorRef}
          onChange={(e) => {
            modelConfigRef.current.backend = e.target.value;
            loadModel();
          }}
          disabled={activeFeature !== null}
          className="ml-2"
        >
          <option value="wasm">Wasm(cpu)</option>
          <option value="webgpu">webGPU</option>
        </select>
      </div>
      <div id="selector-container">
        <label>Model:</label>
        <select
          name="model-selector"
          ref={modelSelectorRef}
          onChange={(e) => {
            modelConfigRef.current.model = e.target.value;
            loadModel();
          }}
          disabled={activeFeature !== null}
          className="ml-2"
        >
          <option value="yolo11n">yolo11n-2.6M</option>
          <option value="yolo11s">yolo11s-9.4M</option>
          {/* <option value="your-custom-model">Your Custom Model</option> */}
          {customModels.map((model, index) => (
            <option key={index} value={model.url}>
              {model.name}
            </option>
          ))}
        </select>
      </div>
      <div>
        <label>Task:</label>
        <select
          name="task-selector"
          ref={taskSelectorRef}
          onChange={(e) => {
            modelConfigRef.current.task = e.target.value;
            loadModel();
          }}
          disabled={activeFeature !== null}
          className="ml-2"
        >
          <option value="detect">Obj detect</option>
          <option value="pose">Pose estimat</option>
          <option value="segment">Segment</option>
        </select>
      </div>
      <div id="selector-container">
        <label>Camera:</label>
        <select
          ref={cameraSelectorRef}
          disabled={activeFeature !== null}
          className="ml-2"
        >
          {cameras.map((camera, index) => (
            <option key={index} value={camera.deviceId}>
              {camera.label || `Camera ${index + 1}`}
            </option>
          ))}
        </select>
      </div>
      <div id="selector-container">
        <label>Imgsz_type:</label>
        <select
          disabled={activeFeature !== null}
          ref={imgszTypeSelectorRef}
          onChange={(e) => {
            modelConfigRef.current.imgsz_type = e.target.value;
          }}
          className="ml-2"
        >
          <option value="dynamic">Dynamic</option>
          <option value="zeroPad">Zero Pad</option>
        </select>
      </div>
    </div>
  );
}

// Display Components
function ImageDisplay({
  cameraRef,
  imgRef,
  overlayRef,
  imgSrc,
  onCameraLoad,
  onImageLoad,
  activeFeature,
}) {
  return (
    <div className="container bg-stone-700 shadow-lg relative min-h-[320px] flex justify-center items-center">
      <video
        className="block md:max-w-[720px] max-h-[640px] rounded-lg mx-auto"
        ref={cameraRef}
        onLoadedMetadata={onCameraLoad}
        hidden={activeFeature !== "camera"}
        autoPlay
      />
      <img
        id="img"
        ref={imgRef}
        src={imgSrc}
        onLoad={onImageLoad}
        hidden={activeFeature !== "image"}
        className="block md:max-w-[720px] max-h-[640px] rounded-lg"
      />
      <canvas
        ref={overlayRef}
        hidden={activeFeature === null}
        className="absolute"
      ></canvas>
    </div>
  );
}

// button Components
function ControlButtons({
  imgSrc,
  fileVideoRef,
  fileImageRef,
  handle_OpenVideo,
  handle_OpenImage,
  handle_ToggleCamera,
  handle_AddModel,
  activeFeature,
}) {
  return (
    <div id="btn-container" className="container flex justify-around gap-x-4">
      <input
        type="file"
        accept="video/mp4"
        hidden
        ref={fileVideoRef}
        onChange={(e) => {
          if (e.target.files[0]) {
            handle_OpenVideo(e.target.files[0]);
            e.target.value = null;
          }
        }}
      />

      <button
        className="btn"
        onClick={() => fileVideoRef.current.click()}
        disabled={activeFeature !== null}
      >
        Open video
      </button>

      <input
        type="file"
        accept="image/*"
        hidden
        ref={fileImageRef}
        onChange={(e) => {
          if (e.target.files[0]) {
            const file = e.target.files[0];
            const imgUrl = URL.createObjectURL(file);
            handle_OpenImage(imgUrl);
            e.target.value = null;
          }
        }}
      />

      <button
        className="btn"
        onClick={() =>
          imgSrc ? handle_OpenImage() : fileImageRef.current.click()
        }
        disabled={activeFeature !== null && activeFeature !== "image"}
      >
        {activeFeature === "image" ? "Close Image" : "Open Image"}
      </button>

      <button
        className="btn"
        onClick={handle_ToggleCamera}
        disabled={activeFeature !== null && activeFeature !== "camera"}
      >
        {activeFeature === "camera" ? "Close Camera" : "Open Camera"}
      </button>

      <label className="btn">
        <input type="file" accept=".onnx" onChange={handle_AddModel} hidden />
        <span>Add model</span>
      </label>
    </div>
  );
}

// model status Components
function ModelStatus({ warnUpTime, inferenceTime, statusMsg, statusColor }) {
  return (
    <div id="model-status-container" className="text-xl md:text-2xl px-2">
      <div
        id="inferenct-time-container"
        className="flex flex-col md:flex-row md:justify-evenly text-lg md:text-xl my-4 md:my-6"
      >
        <p className="mb-2 md:mb-0">
          Warm up time: <span className="text-lime-500">{warnUpTime}ms</span>
        </p>
        <p>
          Inference time:{" "}
          <span className="text-lime-500">{inferenceTime}ms</span>
        </p>
      </div>
      <p
        className={statusColor !== "green" ? "animate-text-loading" : ""}
        style={{ color: statusColor }}
      >
        {statusMsg}
      </p>
    </div>
  );
}

function ResultsTable({ details }) {
  if (details.length === 0) {
    return (
      <details className="text-gray-200 group px-2">
        <summary className="my-2 hover:text-gray-400 cursor-pointer transition-colors duration-300">
          Detection Results ( {details.length} )
        </summary>
        <div className="transition-all duration-300 ease-in-out transform origin-top group-open:animate-details-show">
          <p className="text-center text-gray-400 py-2">No object detected</p>
        </div>
      </details>
    );
  }

  return (
    <details className="text-gray-200 group px-2">
      <summary className="my-5 hover:text-gray-400 cursor-pointer transition-colors duration-300">
        Detection Results ( {details.length} )
      </summary>
      <div
        className="transition-all duration-300 ease-in-out transform origin-top
                group-open:animate-details-show"
      >
        <table
          className="text-left responsive-table mx-auto border-collapse table-auto text-sm 
              bg-gray-800 rounded-md overflow-hidden"
        >
          <thead className="bg-gray-700">
            <tr>
              <th className="border-b border-gray-600 p-2 md:p-4 text-gray-100">
                ID
              </th>
              <th className="border-b border-gray-600 p-2 md:p-4 text-gray-100">
                ClassName
              </th>
              <th className="border-b border-gray-600 p-2 md:p-4 text-gray-100">
                Confidence
              </th>
            </tr>
          </thead>
          <tbody>
            {details.map((item, index) => (
              <tr
                key={index}
                className="hover:bg-gray-700 transition-colors text-gray-300"
              >
                <td className="border-b border-gray-600 p-2 md:p-4">
                  {index}
                </td>
                <td className="border-b border-gray-600 p-2 md:p-4">
                  {classes.class[item.class_idx]}
                </td>
                <td className="border-b border-gray-600 p-2 md:p-4">
                  {(item.score * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

function App() {
  const [modelState, setModelState] = useState({
    warnUpTime: 0,
    inferenceTime: 0,
    statusMsg: "Model not loaded",
    statusColor: "inherit",
  });
  const { warnUpTime, inferenceTime, statusMsg, statusColor } = modelState;

  const modelConfigRef = useRef(MODEL_CONFIG);

  // resource reference
  const backendSelectorRef = useRef(null);
  const modelSelectorRef = useRef(null);
  const taskSelectorRef = useRef(null);
  const cameraSelectorRef = useRef(null);
  const imgszTypeSelectorRef = useRef(null);
  const sessionRef = useRef(null);
  const modelCache = useRef({});

  // content reference
  const imgRef = useRef(null);
  const overlayRef = useRef(null);
  const cameraRef = useRef(null);
  const fileImageRef = useRef(null);
  const fileVideoRef = useRef(null);

  // state
  const [customModels, setCustomModels] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [imgSrc, setImgSrc] = useState(null);
  const [details, setDetails] = useState([]);
  const [activeFeature, setActiveFeature] = useState(null); // null, 'video', 'image', 'camera'

  // worker
  const videoWorkerRef = useRef(null);

  // Init page
  useEffect(() => {
    loadModel();
    getCameras();

    videoWorkerRef.current = new Worker(
      new URL("./utils/video_process_worker.js", import.meta.url),
      {
        type: "module",
      }
    );
    videoWorkerRef.current.onmessage = (e) => {
      setModelState((prev) => ({
        ...prev,
        statusMsg: e.data.statusMsg,
      }));
      if (e.data.processedVideo) {
        const url = URL.createObjectURL(e.data.processedVideo);
        const a = document.createElement("a");
        a.href = url;
        a.download = "processed_video.mp4";
        a.click();
        URL.revokeObjectURL(url);
        setActiveFeature(null);
      }
    };
  }, []);

  const loadModel = useCallback(async () => {
    // Update model state
    setModelState((prev) => ({
      ...prev,
      statusMsg: "Loading model...",
      statusColor: "red",
    }));
    setActiveFeature("loading");

    const modelConfig = modelConfigRef.current;

    // Get model path
    const customModel = customModels.find(
      (model) => model.url === modelConfig.model
    );
    const model_path = customModel
      ? customModel.url
      : `${window.location.href}/models/${modelConfig.model}-${modelConfig.task}.onnx`;
    modelConfig.model_path = model_path;

    const cacheKey = `${modelConfig.model}-${modelConfig.task}-${modelConfig.backend}`;
    if (modelCache.current[cacheKey]) {
      sessionRef.current = modelCache.current[cacheKey];
      setModelState((prev) => ({
        ...prev,
        statusMsg: "Model loaded from cache",
        statusColor: "green",
      }));
      setActiveFeature(null);
      return;
    }

    try {
      // Load model
      const start = performance.now();
      const yolo_model = await model_loader(model_path, modelConfig.backend);
      const end = performance.now();

      sessionRef.current = yolo_model;
      modelCache.current[cacheKey] = yolo_model;

      setModelState((prev) => ({
        ...prev,
        statusMsg: "Model loaded",
        statusColor: "green",
        warnUpTime: (end - start).toFixed(2),
      }));
    } catch (error) {
      setModelState((prev) => ({
        ...prev,
        statusMsg: "Model loading failed",
        statusColor: "red",
      }));
      console.error(error);
    } finally {
      setActiveFeature(null);
    }
  }, [customModels]);

  const getCameras = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(
        (device) => device.kind === "videoinput"
      );
      setCameras(videoDevices);
    } catch (err) {
      console.error("Error getting cameras:", err);
    }
  }, []);

  const handle_AddModel = useCallback((event) => {
    const file = event.target.files[0];
    if (file) {
      const fileName = file.name.replace(".onnx", "");
      const fileUrl = URL.createObjectURL(file);
      setCustomModels((prevModels) => [
        ...prevModels,
        { name: fileName, url: fileUrl },
      ]);
    }
  }, []);

  const handle_OpenImage = useCallback(
    (imgUrl = null) => {
      if (imgUrl) {
        setImgSrc(imgUrl);
        setActiveFeature("image");
      } else if (imgSrc) {
        if (imgSrc.startsWith("blob:")) {
          URL.revokeObjectURL(imgSrc);
        }
        overlayRef.current.width = 0;
        overlayRef.current.height = 0;
        setImgSrc(null);
        setDetails([]);
        setActiveFeature(null);
      }
    },
    [imgSrc]
  );

  const handle_ImageLoad = useCallback(async () => {
    overlayRef.current.width = imgRef.current.width;
    overlayRef.current.height = imgRef.current.height;

    try {
      const src_mat = cv.imread(imgRef.current);
      const [results, results_inferenceTime] = await inference_pipeline(
        src_mat,
        sessionRef.current,
        [overlayRef.current.width, overlayRef.current.height],
        modelConfigRef.current
      );
      const overlayCtx = overlayRef.current.getContext("2d");
      overlayCtx.clearRect(
        0,
        0,
        overlayCtx.canvas.width,
        overlayCtx.canvas.height
      );

      draw_bounding_boxes(results, modelConfigRef.current.task, overlayCtx);
      setDetails(results.bbox_results);
      setModelState((prev) => ({
        ...prev,
        inferenceTime: results_inferenceTime,
      }));
    } catch (error) {
      console.error("Image processing error:", error);
    }
  }, [sessionRef.current]);

  const handle_ToggleCamera = useCallback(async () => {
    if (cameraRef.current.srcObject) {
      // stop camera
      cameraRef.current.srcObject.getTracks().forEach((track) => track.stop());
      cameraRef.current.srcObject = null;
      overlayRef.current.width = 0;
      overlayRef.current.height = 0;

      setDetails([]);
      setActiveFeature(null);
    } else if (cameraSelectorRef.current && cameraSelectorRef.current.value) {
      try {
        // open camera
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: cameraSelectorRef.current.value,
          },
          audio: false,
        });
        cameraRef.current.srcObject = stream;
        setActiveFeature("camera");
      } catch (err) {
        console.error("Error accessing camera:", err);
      }
    }
  }, []);

  const handle_cameraLoad = useCallback(() => {
    overlayRef.current.width = cameraRef.current.clientWidth;
    overlayRef.current.height = cameraRef.current.clientHeight;

    let inputCanvas = new OffscreenCanvas(
      cameraRef.current.videoWidth,
      cameraRef.current.videoHeight
    );
    let ctx = inputCanvas.getContext("2d", {
      willReadFrequently: true,
    });

    const handle_frame_continuous = async () => {
      if (!cameraRef.current?.srcObject) {
        inputCanvas = null;
        ctx = null;
        return;
      }
      ctx.drawImage(
        cameraRef.current,
        0,
        0,
        cameraRef.current.videoWidth,
        cameraRef.current.videoHeight
      ); // draw camera frame to input canvas
      const src_mat = cv.imread(inputCanvas);
      const [results, results_inferenceTime] = await inference_pipeline(
        src_mat,
        sessionRef.current,
        [overlayRef.current.width, overlayRef.current.height],
        modelConfigRef.current
      );
      const overlayCtx = overlayRef.current.getContext("2d");
      overlayCtx.clearRect(
        0,
        0,
        overlayCtx.canvas.width,
        overlayCtx.canvas.height
      );
      draw_bounding_boxes(results, modelConfigRef.current.task, overlayCtx);

      setDetails(results.bbox_results);
      setModelState((prev) => ({
        ...prev,
        inferenceTime: results_inferenceTime,
      }));

      requestAnimationFrame(handle_frame_continuous);
    };
    requestAnimationFrame(handle_frame_continuous);
  }, [sessionRef.current]);

  const handle_OpenVideo = useCallback((file) => {
    if (file) {
      videoWorkerRef.current.postMessage(
        {
          file: file,
          modelConfig: modelConfigRef.current,
        },
        []
      );
      setActiveFeature("video");
    } else {
      setActiveFeature(null);
    }
  }, []);

  return (
    <>
      <h1 className="my-4 md:my-8 text-3xl md:text-4xl px-2">
        Yolo multi object tracking onnx web
      </h1>

      <SettingsPanel
        backendSelectorRef={backendSelectorRef}
        modelSelectorRef={modelSelectorRef}
        taskSelectorRef={taskSelectorRef}
        cameraSelectorRef={cameraSelectorRef}
        imgszTypeSelectorRef={imgszTypeSelectorRef}
        modelConfigRef={modelConfigRef}
        cameras={cameras}
        customModels={customModels}
        loadModel={loadModel}
        activeFeature={activeFeature}
      />

      <ImageDisplay
        cameraRef={cameraRef}
        imgRef={imgRef}
        overlayRef={overlayRef}
        imgSrc={imgSrc}
        onCameraLoad={handle_cameraLoad}
        onImageLoad={handle_ImageLoad}
        activeFeature={activeFeature}
      />

      <ControlButtons
        cameras={cameras}
        imgSrc={imgSrc}
        fileVideoRef={fileVideoRef}
        fileImageRef={fileImageRef}
        handle_OpenVideo={handle_OpenVideo}
        handle_OpenImage={handle_OpenImage}
        handle_ToggleCamera={handle_ToggleCamera}
        handle_AddModel={handle_AddModel}
        activeFeature={activeFeature}
      />

      <ModelStatus
        warnUpTime={warnUpTime}
        inferenceTime={inferenceTime}
        statusMsg={statusMsg}
        statusColor={statusColor}
      />

      <ResultsTable details={details} />
    </>
  );
}

export default App;

import "./assets/App.css";
import classes from "./utils/yolo_classes.json";
import { useEffect, useRef, useState, useCallback } from "react";
import { model_loader } from "./utils/model_loader";
import { inference_pipeline } from "./utils/inference_pipeline";
import { draw_bounding_boxes } from "./utils/draw_bounding_boxes";

// TODO: add support phone screen

const DEFAULT_CONFIG = {
  input_shape: [1, 3, 640, 640],
  iou_threshold: 0.35,
  score_threshold: 0.45,
  backend: "webgpu",
  task: "detect",
};

// set Components
function SettingsPanel({
  backendRef,
  modelRef,
  taskRef,
  cameraSelectorRef,
  cameras,
  customModels,
  onModelChange,
  isModelLoaded,
}) {
  return (
    <div
      id="setting-container"
      className="container text-lg flex flex-col md:flex-row md:justify-evenly gap-2 md:gap-6"
    >
      <div
        id="device-selector-container"
        className="flex items-center justify-between md:justify-start"
      >
        <label htmlFor="device-selector">Backend:</label>
        <select
          name="device-selector"
          ref={backendRef}
          onChange={onModelChange}
          disabled={!isModelLoaded}
          className="ml-2"
        >
          <option value="wasm">Wasm(cpu)</option>
          <option value="webgpu">webGPU</option>
        </select>
      </div>
      <div
        id="model-selector-container"
        className="flex items-center justify-between md:justify-start"
      >
        <label htmlFor="model-selector">Model:</label>
        <select
          name="model-selector"
          ref={modelRef}
          onChange={onModelChange}
          className="ml-2"
        >
          <option value="yolo11n">yolo11n-2.6M</option>
          <option value="yolo11s">yolo11s-9.4M</option>
          {customModels.map((model, index) => (
            <option key={index} value={model.url}>
              {model.name}
            </option>
          ))}
        </select>
      </div>
      <div className="flex items-center justify-between md:justify-start">
        <label htmlFor="task-selector">Task:</label>
        <select
          name="task-selector"
          ref={taskRef}
          onChange={onModelChange}
          className="ml-2"
        >
          <option value="detect">Object detection</option>
          <option value="pose">Pose estimation</option>
        </select>
      </div>
      <div
        id="camera-selector-container"
        className="flex items-center justify-between md:justify-start"
      >
        <label htmlFor="camera-selector">Camera:</label>
        <select ref={cameraSelectorRef} className="ml-2">
          {cameras.map((camera, index) => (
            <option key={index} value={camera.deviceId}>
              {camera.label || `Camera ${index + 1}`}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

// Display Components
function ImageDisplay({
  inputCanvasRef,
  cameraRef,
  imgRef,
  overlayRef,
  imgSrc,
  camera_stream,
  onCameraLoad,
  onImageLoad,
  isProcessing,
}) {
  return (
    <div className="container bg-stone-700 shadow-lg relative min-h-[320px] flex justify-center items-center">
      <canvas ref={inputCanvasRef} hidden></canvas>
      <video
        className="block w-full max-w-full md:max-w-[720px] max-h-[640px] rounded-lg inset-0 mx-auto"
        ref={cameraRef}
        onLoadedData={onCameraLoad}
        hidden={!camera_stream}
        autoPlay
      />
      <img
        id="img"
        ref={imgRef}
        src={imgSrc}
        onLoad={onImageLoad}
        hidden={camera_stream}
        className="block inset-0 w-full max-w-full md:max-w-[720px] max-h-[640px] rounded-lg"
      />
      <canvas ref={overlayRef} className="absolute"></canvas>
      {isProcessing && (
        <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center text-white">
          <div className="animate-pulse text-xl">Processing...</div>
        </div>
      )}
    </div>
  );
}

// button Components
function ControlButtons({
  camera_stream,
  cameras,
  imgSrc,
  isModelLoaded,
  openImageRef,
  onOpenImageClick,
  onToggleCamera,
  onAddModel,
}) {
  return (
    <div id="btn-container" className="container flex justify-around gap-x-4">
      <input
        type="file"
        accept="image/*"
        hidden
        ref={openImageRef}
        onChange={(e) => {
          if (e.target.files[0]) {
            const file = e.target.files[0];
            const imgUrl = URL.createObjectURL(file);
            onOpenImageClick(imgUrl);
            e.target.value = null;
          }
        }}
      />

      <button
        className="btn"
        disabled={camera_stream || !isModelLoaded}
        onClick={() =>
          imgSrc ? onOpenImageClick() : openImageRef.current.click()
        }
      >
        {imgSrc ? "Close Image" : "Open Image"}
      </button>

      <button
        className="btn"
        onClick={onToggleCamera}
        disabled={cameras.length === 0 || imgSrc || !isModelLoaded}
      >
        {camera_stream ? "Close Camera" : "Open Camera"}
      </button>

      <label className="btn cursor-pointer">
        <input type="file" accept=".onnx" onChange={onAddModel} hidden />
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
  return (
    <details className="text-gray-200 group px-2">
      <summary className="my-5 hover:text-gray-400 cursor-pointer transition-colors duration-300">
        Detected objects
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
                Number
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
                  {index + 1}
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
    isLoaded: false,
    warnUpTime: 0,
    inferenceTime: 0,
    statusMsg: "Model not loaded",
    statusColor: "inherit",
  });
  const {
    isLoaded: isModelLoaded,
    warnUpTime,
    inferenceTime,
    statusMsg,
    statusColor,
  } = modelState;

  // resource reference
  const modelCache = useRef({});
  const canvasContextRef = useRef(null);
  const backendRef = useRef(null);
  const modelRef = useRef(null);
  const taskRef = useRef(null);
  const cameraSelectorRef = useRef(null);
  const sessionRef = useRef(null);

  // content reference
  const imgRef = useRef(null);
  const overlayRef = useRef(null);
  const cameraRef = useRef(null);
  const inputCanvasRef = useRef(null);
  const openImageRef = useRef(null);

  // state
  const [customModels, setCustomModels] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [camera_stream, setCameraStream] = useState(null);
  const [imgSrc, setImgSrc] = useState(null);
  const [details, setDetails] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [config, setConfig] = useState({ ...DEFAULT_CONFIG });

  // init
  useEffect(() => {
    loadModel();
    getCameras();

    return () => {
      // cleanup
      if (camera_stream) {
        camera_stream.getTracks().forEach((track) => track.stop());
      }

      // cleanup custom models
      customModels.forEach((model) => {
        if (model.url && model.url.startsWith("blob:")) {
          URL.revokeObjectURL(model.url);
        }
      });

      if (imgSrc && imgSrc.startsWith("blob:")) {
        URL.revokeObjectURL(imgSrc);
      }
    };
  }, []);

  // init canvas context
  useEffect(() => {
    if (inputCanvasRef.current) {
      canvasContextRef.current = inputCanvasRef.current.getContext("2d", {
        willReadFrequently: true,
      });
    }
  }, []);

  const loadModel = useCallback(async () => {
    // update model state
    setModelState((prev) => ({
      ...prev,
      statusMsg: "Loading model...",
      statusColor: "red",
      isLoaded: false,
    }));

    // get model config
    const backend = backendRef.current?.value || "webgpu";
    const task = taskRef.current?.value || "detect";
    const selectedModel = modelRef.current?.value || "yolo11n";

    // update config
    const newConfig = { ...DEFAULT_CONFIG, backend, task };
    setConfig(newConfig);

    const customModel = customModels.find(
      (model) => model.url === selectedModel
    );

    const model_path = customModel
      ? customModel.url
      : `${window.location.href}/models/${selectedModel}-${task}-simplify-dynamic.onnx`;

    const cacheKey = `${selectedModel}-${task}-${backend}`;
    if (modelCache.current[cacheKey]) {
      sessionRef.current = modelCache.current[cacheKey];
      setModelState((prev) => ({
        ...prev,
        statusMsg: "Model loaded from cache",
        statusColor: "green",
        isLoaded: true,
      }));
      return;
    }

    try {
      // load model
      const start = performance.now();
      const yolo_model = await model_loader(model_path, newConfig);
      const end = performance.now();

      sessionRef.current = yolo_model;
      modelCache.current[cacheKey] = yolo_model;

      setModelState((prev) => ({
        ...prev,
        statusMsg: "Model loaded",
        statusColor: "green",
        warnUpTime: (end - start).toFixed(2),
        isLoaded: true,
      }));
    } catch (error) {
      setModelState((prev) => ({
        ...prev,
        statusMsg: "Model loading failed",
        statusColor: "red",
        isLoaded: false,
      }));
      console.error(error);
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
      } else if (imgSrc) {
        if (imgSrc.startsWith("blob:")) {
          URL.revokeObjectURL(imgSrc);
        }
        setImgSrc("");
        if (overlayRef.current) {
          overlayRef.current.width = 0;
          overlayRef.current.height = 0;
        }
        setDetails([]);
      }
    },
    [imgSrc]
  );

  const handle_ImageLoad = useCallback(async () => {
    if (!imgRef.current || !overlayRef.current || !sessionRef.current) return;

    setIsProcessing(true);
    overlayRef.current.width = imgRef.current.width;
    overlayRef.current.height = imgRef.current.height;

    try {
      const [results, results_inferenceTime] = await inference_pipeline(
        imgRef.current,
        sessionRef.current,
        config
      );
      setDetails(results);
      setModelState((prev) => ({
        ...prev,
        inferenceTime: results_inferenceTime,
      }));
      draw_bounding_boxes(results, config.task, overlayRef.current);
    } catch (error) {
      console.error("Image processing error:", error);
    } finally {
      setIsProcessing(false);
    }
  }, [config, sessionRef.current]);

  const handle_ToggleCamera = useCallback(async () => {
    if (camera_stream) {
      // 停止相機
      camera_stream.getTracks().forEach((track) => track.stop());
      cameraRef.current.srcObject = null;
      setCameraStream(null);
      if (overlayRef.current) {
        overlayRef.current.width = 0;
        overlayRef.current.height = 0;
      }
      setDetails([]);
    } else if (cameraSelectorRef.current && cameraSelectorRef.current.value) {
      try {
        // 啟動相機
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: cameraSelectorRef.current.value,
          },
          audio: false,
        });
        setCameraStream(stream);
        cameraRef.current.srcObject = stream;
      } catch (err) {
        console.error("Error accessing camera:", err);
      }
    }
  }, [camera_stream]);

  const handle_cameraLoad = useCallback(() => {
    if (
      !cameraRef.current ||
      !canvasContextRef.current ||
      !inputCanvasRef.current ||
      !overlayRef.current
    )
      return;
    const ctx = canvasContextRef.current;

    // set input canvas
    ctx.canvas.width = cameraRef.current.videoWidth;
    ctx.canvas.height = cameraRef.current.videoHeight;

    // set screen overlay
    const videoRect = cameraRef.current.getBoundingClientRect();
    overlayRef.current.width = videoRect.width;
    overlayRef.current.height = videoRect.height;

    console.log(cameraRef.current.videoWidth, cameraRef.current.videoHeight);

    handle_frame_continuous(ctx);
  }, [sessionRef.current]);

  const handle_frame_continuous = useCallback(
    async (ctx) => {
      if (!cameraRef.current?.srcObject) return;

      // 30fps
      const now = performance.now();
      if (!window.lastFrameTime || now - window.lastFrameTime > 33) {
        window.lastFrameTime = now;

        // Render frame to canvas
        ctx.drawImage(
          cameraRef.current,
          0,
          0,
          cameraRef.current.videoWidth,
          cameraRef.current.videoHeight
        );

        try {
          const [results, results_inferenceTime] = await inference_pipeline(
            inputCanvasRef.current,
            sessionRef.current,
            config
          );

          // only update state if results change to reduce re-render
          if (JSON.stringify(results) !== JSON.stringify(details)) {
            setDetails(results);
          }

          // only update state if inference time changes to reduce re-render
          if (results_inferenceTime !== inferenceTime) {
            setModelState((prev) => ({
              ...prev,
              inferenceTime: results_inferenceTime,
            }));
          }

          draw_bounding_boxes(results, config.task, overlayRef.current);
        } catch (error) {
          console.error("Frame processing error:", error);
        }
      }

      // next frame
      requestAnimationFrame(() => handle_frame_continuous(ctx));
    },
    [config, sessionRef.current, details, inferenceTime]
  );

  return (
    <>
      <h1 className="my-4 md:my-8 text-3xl md:text-4xl px-2">
        Yolo multi task onnx web
      </h1>

      <SettingsPanel
        backendRef={backendRef}
        modelRef={modelRef}
        taskRef={taskRef}
        cameraSelectorRef={cameraSelectorRef}
        cameras={cameras}
        customModels={customModels}
        onModelChange={loadModel}
        isModelLoaded={isModelLoaded}
      />

      <ImageDisplay
        inputCanvasRef={inputCanvasRef}
        cameraRef={cameraRef}
        imgRef={imgRef}
        overlayRef={overlayRef}
        imgSrc={imgSrc}
        camera_stream={camera_stream}
        onCameraLoad={handle_cameraLoad}
        onImageLoad={handle_ImageLoad}
        isProcessing={isProcessing}
      />

      <ControlButtons
        camera_stream={camera_stream}
        cameras={cameras}
        imgSrc={imgSrc}
        isModelLoaded={isModelLoaded}
        openImageRef={openImageRef}
        onOpenImageClick={handle_OpenImage}
        onToggleCamera={handle_ToggleCamera}
        onAddModel={handle_AddModel}
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

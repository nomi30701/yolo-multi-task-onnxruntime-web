import "./assets/App.css";
import { useEffect, useRef, useState, useCallback } from "react";
import { model_loader } from "./utils/model_loader";
import { inference_pipeline } from "./utils/inference_pipeline";
import { render_overlay } from "./utils/render_overlay";
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

// Settings Components
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
      className="container bg-gray-800 rounded-xl shadow-lg p-3 sm:p-4 mb-4 sm:mb-6 overflow-hidden"
    >
      <h2 className="text-lg sm:text-xl font-bold mb-3 text-gray-200 border-b border-gray-700 pb-2">
        Model Settings
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3">
        {/* Settings items */}
        <div className="flex flex-col">
          <label className="text-gray-300 mb-1 text-sm sm:text-base font-medium">
            Backend:
          </label>
          <select
            name="device-selector"
            ref={backendSelectorRef}
            onChange={(e) => {
              modelConfigRef.current.backend = e.target.value;
              loadModel();
            }}
            disabled={activeFeature !== null}
            className="p-2 text-sm rounded-md bg-gray-700 text-white border border-gray-600 focus:border-violet-500 focus:ring-2 focus:ring-violet-500"
          >
            <option value="wasm">Wasm(cpu)</option>
            <option value="webgpu">webGPU</option>
          </select>
        </div>
        <div className="flex flex-col">
          <label className="text-gray-300 mb-1 font-medium">Model:</label>
          <select
            name="model-selector"
            ref={modelSelectorRef}
            onChange={(e) => {
              modelConfigRef.current.model = e.target.value;
              loadModel();
            }}
            disabled={activeFeature !== null}
            className="p-2 rounded-md bg-gray-700 text-white border border-gray-600 focus:border-violet-500 focus:ring-2 focus:ring-violet-500"
          >
            <option value="yolo11n">yolo11n-2.6M</option>
            <option value="yolo11s">yolo11s-9.4M</option>
            <option value="yolo12n">yolo12n-2.6M</option>
            <option value="yolo12s">yolo12s-9.3M</option>
            {customModels.map((model, index) => (
              <option key={index} value={model.url}>
                {model.name}
              </option>
            ))}
          </select>
        </div>
        <div className="flex flex-col">
          <label className="text-gray-300 mb-1 font-medium">Task:</label>
          <select
            name="task-selector"
            ref={taskSelectorRef}
            onChange={(e) => {
              modelConfigRef.current.task = e.target.value;
              loadModel();
            }}
            disabled={activeFeature !== null}
            className="p-2 rounded-md bg-gray-700 text-white border border-gray-600 focus:border-violet-500 focus:ring-2 focus:ring-violet-500"
          >
            <option value="detect">Object Detection</option>
            <option value="pose">Pose Estimation</option>
            <option value="segment">Segmentation</option>
          </select>
        </div>
        <div className="flex flex-col">
          <label className="text-gray-300 mb-1 font-medium">Camera:</label>
          <select
            ref={cameraSelectorRef}
            disabled={activeFeature !== null}
            className="p-2 rounded-md bg-gray-700 text-white border border-gray-600 focus:border-violet-500 focus:ring-2 focus:ring-violet-500"
          >
            {cameras.map((camera, index) => (
              <option key={index} value={camera.deviceId}>
                {camera.label || `Camera ${index + 1}`}
              </option>
            ))}
          </select>
        </div>
        <div className="flex flex-col">
          <label className="text-gray-300 mb-1 font-medium">Image Type:</label>
          <select
            disabled={activeFeature !== null}
            ref={imgszTypeSelectorRef}
            onChange={(e) => {
              modelConfigRef.current.imgsz_type = e.target.value;
            }}
            className="p-2 rounded-md bg-gray-700 text-white border border-gray-600 focus:border-violet-500 focus:ring-2 focus:ring-violet-500"
          >
            <option value="dynamic">Dynamic</option>
            <option value="zeroPad">Zero Pad</option>
          </select>
        </div>
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
    <div className="container bg-gray-800 rounded-xl shadow-lg relative min-h-[200px] sm:min-h-[320px] flex justify-center items-center overflow-hidden p-2 md:p-4 mb-4 sm:mb-6">
      {activeFeature === null && (
        <div className="text-gray-400 text-center p-4 sm:p-8">
          <svg
            className="w-16 h-16 sm:w-24 sm:h-24 mx-auto mb-2 sm:mb-4 text-gray-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"
            />
          </svg>
          <p className="text-base sm:text-xl font-medium">No media selected</p>
          <p className="mt-1 sm:mt-2 text-xs sm:text-base">
            Use the controls below to open an image, video, or camera
          </p>
        </div>
      )}
      <video
        className="block max-h-[400px] sm:max-h-[640px] rounded-lg mx-auto object-contain"
        ref={cameraRef}
        onLoadedMetadata={onCameraLoad}
        hidden={activeFeature !== "camera"}
        autoPlay
        playsInline
      />
      <img
        id="img"
        ref={imgRef}
        src={imgSrc}
        onLoad={onImageLoad}
        hidden={activeFeature !== "image"}
        className="block max-h-[400px] sm:max-h-[640px] rounded-lg mx-auto object-contain"
        alt="Uploaded"
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
  handle_AddClassesFile,
  customClasses,
  currentClasses,
  setCurrentClasses,
  activeFeature,
}) {
  return (
    <div className="container bg-gray-800 rounded-xl shadow-lg p-3 sm:p-4 mb-4 sm:mb-6">
      <div className="grid grid-cols-2 gap-2 sm:gap-3">
        {/* Input and buttons */}
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
          className="btn-primary flex items-center justify-center text-sm sm:text-base"
          onClick={() => fileVideoRef.current.click()}
          disabled={activeFeature !== null}
        >
          <svg
            className="w-4 h-4 sm:w-5 sm:h-5 mr-1 sm:mr-2"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
            />
          </svg>
          <span className="truncate">Open Video</span>
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
          className={`${
            activeFeature === "image" ? "btn-danger" : "btn-primary"
          } flex items-center justify-center`}
          onClick={() =>
            imgSrc ? handle_OpenImage() : fileImageRef.current.click()
          }
          disabled={activeFeature !== null && activeFeature !== "image"}
        >
          {activeFeature === "image" ? (
            <>
              <svg
                className="w-5 h-5 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
              Close Image
            </>
          ) : (
            <>
              <svg
                className="w-5 h-5 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
              Open Image
            </>
          )}
        </button>

        <button
          className={`${
            activeFeature === "camera" ? "btn-danger" : "btn-primary"
          } flex items-center justify-center`}
          onClick={handle_ToggleCamera}
          disabled={activeFeature !== null && activeFeature !== "camera"}
        >
          {activeFeature === "camera" ? (
            <>
              <svg
                className="w-5 h-5 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
              Close Camera
            </>
          ) : (
            <>
              <svg
                className="w-5 h-5 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"
                />
              </svg>
              Open Camera
            </>
          )}
        </button>

        <button
          className="btn-secondary flex items-center justify-center"
          onClick={(e) => {
            const input = document.createElement("input");
            input.type = "file";
            input.accept = ".onnx";
            input.onchange = handle_AddModel;
            input.click();
          }}
          disabled={activeFeature !== null}
        >
          <svg
            className="w-5 h-5 mr-2"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          Add Model
        </button>

        <button
          className="btn-secondary flex items-center justify-center"
          onClick={(e) => {
            const input = document.createElement("input");
            input.type = "file";
            input.accept = ".json";
            input.onchange = handle_AddClassesFile;
            input.click();
          }}
          disabled={activeFeature !== null}
        >
          <svg
            className="w-5 h-5 mr-2"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"
            />
          </svg>
          Add Classes
        </button>
      </div>

      {customClasses.length > 0 && (
        <div className="mt-3 sm:mt-4">
          <div className="flex flex-col">
            <label className="text-gray-300 mb-1 text-sm sm:text-base font-medium">
              Class File:
            </label>
            <select
              className="p-2 text-sm rounded-md bg-gray-700 text-white border border-gray-600 focus:border-violet-500 focus:ring-2 focus:ring-violet-500"
              value={
                currentClasses === classes
                  ? "default"
                  : customClasses.findIndex((c) => c.data === currentClasses)
              }
              onChange={(e) => {
                const value = e.target.value;
                if (value === "default") {
                  setCurrentClasses(classes);
                } else {
                  const index = parseInt(value);
                  setCurrentClasses(customClasses[index].data);
                }
              }}
            >
              <option value="default">Default Classes</option>
              {customClasses.map((classFile, index) => (
                <option key={index} value={index}>
                  {classFile.name}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}
    </div>
  );
}

function ModelStatus({ warnUpTime, inferenceTime, statusMsg, statusColor }) {
  return (
    <div className="container bg-gray-800 rounded-xl shadow-lg p-3 sm:p-4 mb-4 sm:mb-6">
      <h2 className="text-lg sm:text-xl font-bold mb-3 text-gray-200 border-b border-gray-700 pb-2">
        Model Performance
      </h2>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-3">
        <div className="bg-gray-700 p-3 rounded-lg">
          <div className="text-gray-400 text-xs sm:text-sm font-medium mb-1">
            Warm Up Time
          </div>
          <div className="text-xl sm:text-2xl font-bold text-lime-500">
            {warnUpTime} ms
          </div>
        </div>

        <div className="bg-gray-700 p-3 rounded-lg">
          <div className="text-gray-400 text-xs sm:text-sm font-medium mb-1">
            Inference Time
          </div>
          <div className="text-xl sm:text-2xl font-bold text-lime-500">
            {inferenceTime} ms
          </div>
        </div>
      </div>

      <div className="bg-gray-700 p-3 rounded-lg flex items-center">
        <div className="mr-2 sm:mr-3">
          <div
            className={`w-2 h-2 sm:w-3 sm:h-3 rounded-full ${
              statusColor === "green"
                ? "bg-green-500"
                : statusColor === "red"
                ? "bg-red-500"
                : statusColor === "blue"
                ? "bg-blue-500"
                : "bg-gray-500"
            } ${statusColor !== "green" ? "animate-pulse" : ""}`}
          ></div>
        </div>
        <p
          className={`${
            statusColor !== "green" ? "animate-text-loading" : ""
          } text-sm sm:text-lg`}
          style={{
            color:
              statusColor === "green"
                ? "#10b981"
                : statusColor === "red"
                ? "#ef4444"
                : statusColor === "blue"
                ? "#3b82f6"
                : "#d1d5db",
          }}
        >
          {statusMsg}
        </p>
      </div>
    </div>
  );
}

function ResultsTable({ details }) {
  return (
    <div className="container bg-gray-800 rounded-xl shadow-lg p-3 sm:p-4 mb-4 sm:mb-6">
      <details className="text-gray-200 group">
        <summary className="flex items-center cursor-pointer select-none">
          <div className="flex-1 text-lg sm:text-xl font-bold border-b border-gray-700 pb-2">
            Detection Results ({details.length})
          </div>
          <div className="text-gray-400">
            <svg
              className="w-4 h-4 sm:w-5 sm:h-5 transform group-open:rotate-180 transition-transform duration-200"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 9l-7 7-7-7"
              />
            </svg>
          </div>
        </summary>

        <div className="transition-all duration-300 ease-in-out transform origin-top group-open:animate-details-show mt-3 sm:mt-4">
          {details.length === 0 ? (
            <div className="bg-gray-700 rounded-lg p-4 sm:p-8 text-center">
              <svg
                className="w-8 h-8 sm:w-12 sm:h-12 mx-auto mb-3 sm:mb-4 text-gray-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <p className="text-gray-400 text-base sm:text-lg">
                No objects detected
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto -mx-3 px-3">
              <table className="w-full border-collapse min-w-full">
                <thead>
                  <tr className="bg-gray-700 text-left">
                    <th className="p-2 sm:p-3 rounded-tl-lg text-xs sm:text-sm">
                      ID
                    </th>
                    <th className="p-2 sm:p-3 text-xs sm:text-sm">Class</th>
                    <th className="p-2 sm:p-3 rounded-tr-lg text-xs sm:text-sm">
                      Confidence
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {details.map((item, index) => (
                    <tr
                      key={index}
                      className={`border-b border-gray-700 hover:bg-gray-700 transition-colors text-gray-300 ${
                        index === details.length - 1 ? "border-b-0" : ""
                      }`}
                    >
                      <td className="p-2 sm:p-3 font-mono text-xs sm:text-sm">
                        {index}
                      </td>
                      <td className="p-2 sm:p-3 font-medium text-xs sm:text-sm">
                        {classes.class[item.class_idx]}
                      </td>
                      <td className="p-2 sm:p-3 text-xs sm:text-sm">
                        <div className="flex items-center">
                          <div className="w-full bg-gray-600 rounded-full h-1.5 sm:h-2.5 mr-1 sm:mr-2 max-w-[70px] sm:max-w-[100px]">
                            <div
                              className="bg-lime-500 h-1.5 sm:h-2.5 rounded-full"
                              style={{ width: `${item.score * 100}%` }}
                            ></div>
                          </div>
                          <span>{(item.score * 100).toFixed(1)}%</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </details>
    </div>
  );
}

function App() {
  const [processingStatus, setProcessingStatus] = useState({
    warnUpTime: 0,
    inferenceTime: 0,
    statusMsg: "Model not loaded",
    statusColor: "inherit",
  });

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

  // custom classes
  const [customClasses, setCustomClasses] = useState([]);
  const [currentClasses, setCurrentClasses] = useState(classes);

  // Worker
  const videoWorkerRef = useRef(null);

  // Init page
  useEffect(() => {
    loadModel();

    // Worker setup
    const videoWorker = new Worker(
      new URL("./utils/video_process_worker.js", import.meta.url),
      { type: "module" }
    );

    videoWorker.onmessage = videoWorkerMessage;
    videoWorkerRef.current = videoWorker;
  }, []);

  const videoWorkerMessage = useCallback((e) => {
    setProcessingStatus((prev) => ({
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
  }, []);

  const loadModel = useCallback(async () => {
    // Update model state
    setProcessingStatus((prev) => ({
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
      setProcessingStatus((prev) => ({
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

      setProcessingStatus((prev) => ({
        ...prev,
        statusMsg: "Model loaded",
        statusColor: "green",
        warnUpTime: (end - start).toFixed(2),
      }));
    } catch (error) {
      setProcessingStatus((prev) => ({
        ...prev,
        statusMsg: "Model loading failed",
        statusColor: "red",
      }));
      console.error(error);
    } finally {
      setActiveFeature(null);
    }
  }, [customModels]);

  // Button add model
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

  // Button add classes file
  const handle_AddClassesFile = useCallback((event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const jsonData = JSON.parse(e.target.result);
          // Validate that the JSON has the expected format
          if (
            jsonData &&
            jsonData.class &&
            typeof jsonData.class === "object"
          ) {
            const fileName = file.name.replace(".json", "");
            const newClassFile = { name: fileName, data: jsonData };
            setCustomClasses((prevClasses) => [...prevClasses, newClassFile]);

            // Automatically select the newly added classes
            setCurrentClasses(jsonData);
          } else {
            setProcessingStatus((prev) => ({
              ...prev,
              statusMsg: "Invalid classes file format",
              statusColor: "red",
            }));
          }
        } catch (error) {
          console.error("Error parsing JSON file:", error);
          setProcessingStatus((prev) => ({
            ...prev,
            statusMsg: "Error parsing JSON file",
            statusColor: "red",
          }));
        }
      };
      reader.readAsText(file);
    }
  }, []);

  // Button Upload Image
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

  // If image loaded, run inference
  const handle_ImageLoad = useCallback(async () => {
    // overlay size = image size
    overlayRef.current.width = imgRef.current.width;
    overlayRef.current.height = imgRef.current.height;

    // inference
    try {
      const [results, results_inferenceTime] = await inference_pipeline(
        imgRef.current,
        sessionRef.current,
        [overlayRef.current.width, overlayRef.current.height],
        modelConfigRef.current
      );
      // draw results on overlay
      const overlayCtx = overlayRef.current.getContext("2d");
      overlayCtx.clearRect(
        0,
        0,
        overlayCtx.canvas.width,
        overlayCtx.canvas.height
      );
      await render_overlay(
        results,
        modelConfigRef.current.task,
        overlayCtx,
        currentClasses
      );

      setDetails(results.bbox_results);
      setProcessingStatus((prev) => ({
        ...prev,
        inferenceTime: results_inferenceTime,
      }));
    } catch (error) {
      console.error("Image processing error:", error);
    }
  }, [sessionRef.current]);

  // Get camera list
  const getCameras = useCallback(async () => {
    try {
      // get camera list
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(
        (device) => device.kind === "videoinput"
      );

      // if no labels, try request permission to get labels
      if (videoDevices.length > 0 && !videoDevices[0].label) {
        try {
          const tempStream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false,
          });
          tempStream.getTracks().forEach((track) => track.stop());

          const updatedDevices =
            await navigator.mediaDevices.enumerateDevices();
          const updatedVideoDevices = updatedDevices.filter(
            (device) => device.kind === "videoinput"
          );

          setCameras(updatedVideoDevices);
          return updatedVideoDevices;
        } catch (err) {
          console.error("Error getting camera permissions:", err);
          setCameras(videoDevices);
          return videoDevices;
        }
      } else {
        setCameras(videoDevices);
        return videoDevices;
      }
    } catch (err) {
      console.error("Error enumerating devices:", err);
      setCameras([]);
      return [];
    }
  }, []);

  // Button toggle camera
  const handle_ToggleCamera = useCallback(async () => {
    if (cameraRef.current.srcObject) {
      // close camera
      cameraRef.current.srcObject.getTracks().forEach((track) => track.stop());
      cameraRef.current.srcObject = null;
      overlayRef.current.width = 0;
      overlayRef.current.height = 0;

      setDetails([]);
      setActiveFeature(null);
    } else {
      // open camera
      try {
        setProcessingStatus((prev) => ({
          ...prev,
          statusMsg: "Getting camera list...",
          statusColor: "blue",
        }));
        const currentCameras = await getCameras();
        if (currentCameras.length === 0) {
          throw new Error("No available camera devices found");
        }

        setProcessingStatus((prev) => ({
          ...prev,
          statusMsg: "Opening camera...",
          statusColor: "blue",
        }));

        const selectedDeviceId = cameraSelectorRef.current
          ? cameraSelectorRef.current.value
          : currentCameras[0].deviceId;

        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            video: {
              deviceId: { exact: selectedDeviceId },
            },
            audio: false,
          });

          cameraRef.current.srcObject = stream;
          setActiveFeature("camera");
          setProcessingStatus((prev) => ({
            ...prev,
            statusMsg: "Camera opened successfully",
            statusColor: "green",
          }));
        } catch (streamErr) {
          console.error("Failed to open selected camera:", streamErr);

          setProcessingStatus((prev) => ({
            ...prev,
            statusMsg: "Trying to open any available camera...",
            statusColor: "blue",
          }));

          const fallbackStream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false,
          });

          cameraRef.current.srcObject = fallbackStream;
          setActiveFeature("camera");
          setProcessingStatus((prev) => ({
            ...prev,
            statusMsg: "Default camera opened (selected camera unavailable)",
            statusColor: "green",
          }));
        }
      } catch (err) {
        console.error("Error in camera toggle process:", err);
        setProcessingStatus((prev) => ({
          ...prev,
          statusMsg: `Camera opened failed: ${err.message}`,
          statusColor: "red",
        }));
      }
    }
  }, [getCameras]);

  // If camera loaded, run inference continuously
  const handle_cameraLoad = useCallback(() => {
    overlayRef.current.width = cameraRef.current.clientWidth;
    overlayRef.current.height = cameraRef.current.clientHeight;

    // create offscreen canvas for input
    let inputCanvas = new OffscreenCanvas(
      cameraRef.current.videoWidth,
      cameraRef.current.videoHeight
    );
    let ctx = inputCanvas.getContext("2d", {
      willReadFrequently: true,
    });

    // inference loop
    const handle_frame_continuous = async () => {
      if (!cameraRef.current?.srcObject) {
        inputCanvas = null;
        ctx = null;
        return;
      }
      // draw camera frame to input canvas
      ctx.drawImage(
        cameraRef.current,
        0,
        0,
        cameraRef.current.videoWidth,
        cameraRef.current.videoHeight
      );
      // Inference
      const [results, results_inferenceTime] = await inference_pipeline(
        inputCanvas,
        sessionRef.current,
        [overlayRef.current.width, overlayRef.current.height],
        modelConfigRef.current
      );
      // draw results on overlay
      const overlayCtx = overlayRef.current.getContext("2d");
      overlayCtx.clearRect(
        0,
        0,
        overlayCtx.canvas.width,
        overlayCtx.canvas.height
      );
      render_overlay(
        results,
        modelConfigRef.current.task,
        overlayCtx,
        currentClasses
      );

      setDetails(results.bbox_results);
      setProcessingStatus((prev) => ({
        ...prev,
        inferenceTime: results_inferenceTime,
      }));

      requestAnimationFrame(handle_frame_continuous);
    };
    requestAnimationFrame(handle_frame_continuous);
  }, [sessionRef.current]);

  // Button Upload Video
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
    <div className="max-w-7xl mx-auto px-2 sm:px-4 lg:px-6 py-4 sm:py-6 bg-gray-900 min-h-screen">
      <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold text-center mb-4 sm:mb-6 text-white">
        <span className="block sm:inline">YOLO Multi-Task</span>
        <span className="bg-gradient-to-r from-violet-500 to-fuchsia-500 bg-clip-text text-transparent block sm:inline">
          {" "}
          Object Detection
        </span>
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
        imgSrc={imgSrc}
        fileVideoRef={fileVideoRef}
        fileImageRef={fileImageRef}
        handle_OpenVideo={handle_OpenVideo}
        handle_OpenImage={handle_OpenImage}
        handle_ToggleCamera={handle_ToggleCamera}
        handle_AddModel={handle_AddModel}
        handle_AddClassesFile={handle_AddClassesFile}
        customClasses={customClasses}
        currentClasses={currentClasses}
        setCurrentClasses={setCurrentClasses}
        activeFeature={activeFeature}
      />

      <ModelStatus
        warnUpTime={processingStatus.warnUpTime}
        inferenceTime={processingStatus.inferenceTime}
        statusMsg={processingStatus.statusMsg}
        statusColor={processingStatus.statusColor}
      />

      <ResultsTable details={details} currentClasses={currentClasses} />
    </div>
  );
}

export default App;

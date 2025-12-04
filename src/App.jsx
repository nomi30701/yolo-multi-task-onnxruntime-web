import "./assets/App.css";
import { useEffect, useRef, useState, useCallback } from "react";
import { model_loader } from "./utils/model_loader";
import { inference_pipeline } from "./utils/inference_pipeline";
import { render_overlay } from "./utils/render_overlay";
import classes from "./utils/yolo_classes.json";

// Components
import SettingsPanel from "./components/SettingsPanel";
import ImageDisplay from "./components/ImageDisplay";
import ControlButtons from "./components/ControlButtons";
import ModelStatus from "./components/ModelStatus";
import ResultsTable from "./components/ResultsTable";

const MODEL_CONFIG = {
  input_shape: [1, 3, 640, 640],
  iou_threshold: 0.35,
  score_threshold: 0.45,
  backend: "wasm",
  model: "yolo11n",
  model_path: "",
  task: "detect",
  imgsz_type: "dynamic",
  classes: classes,
};

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
  const classFileSelectedRef = useRef(null);
  // const [currentClasses, setCurrentClasses] = useState(classes);

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
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const jsonData = JSON.parse(e.target.result);

        const fileName = file.name.replace(/\.json$/i, "");
        setCustomClasses((prev) => [
          ...prev,
          { name: fileName, data: jsonData },
        ]);

        setProcessingStatus((prev) => ({
          ...prev,
          statusMsg: `Classes file "${fileName}" loaded successfully`,
          statusColor: "green",
        }));
      } catch (error) {
        console.error("Error parsing JSON file:", error);
        setProcessingStatus((prev) => ({
          ...prev,
          statusMsg: error.message || "Error parsing JSON file",
          statusColor: "red",
        }));
      }
    };

    reader.onerror = () => {
      setProcessingStatus((prev) => ({
        ...prev,
        statusMsg: "Failed to read file",
        statusColor: "red",
      }));
    };

    reader.readAsText(file);
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
        modelConfigRef.current.classes
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
        modelConfigRef.current.classes
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
        customClasses={customClasses}
        classFileSelectedRef={classFileSelectedRef}
        cameras={cameras}
        customModels={customModels}
        loadModel={loadModel}
        activeFeature={activeFeature}
        defaultClasses={classes}
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
        activeFeature={activeFeature}
      />

      <ModelStatus
        warnUpTime={processingStatus.warnUpTime}
        inferenceTime={processingStatus.inferenceTime}
        statusMsg={processingStatus.statusMsg}
        statusColor={processingStatus.statusColor}
      />

      <ResultsTable
        details={details}
        currentClasses={modelConfigRef.current.classes.classes}
      />
    </div>
  );
}

export default App;
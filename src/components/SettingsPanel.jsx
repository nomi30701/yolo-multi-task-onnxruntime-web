import { memo, useRef, useState } from "react";

const SettingsPanel = memo(function SettingsPanel({
  cameraSelectorRef,
  imgszTypeSelectorRef,
  modelConfigRef,
  customClasses,
  cameras,
  customModels,
  activeFeature,
  defaultClasses,
  loadModel,
}) {
  const [nmsEnabled, setNmsEnabled] = useState(true);
  const [nmsLocked, setNmsLocked] = useState(true);
  const threadTimeoutRef = useRef(null);

  const handleThreadChange = (e) => {
    const value = e.target.value;

    if (threadTimeoutRef.current) {
      clearTimeout(threadTimeoutRef.current);
    }

    threadTimeoutRef.current = setTimeout(() => {
      const numThreads = parseInt(value);
      if (!isNaN(numThreads) && numThreads > 0) {
        modelConfigRef.current.numThreads = numThreads;
        loadModel();
      }
    }, 1000);
  };

  const sectionClass =
    "bg-gray-700/50 rounded-lg p-4 border border-gray-600/50";
  const labelClass =
    "text-gray-400 mb-1.5 text-xs uppercase tracking-wider font-semibold";
  const inputClass =
    "w-full p-2.5 text-sm rounded-lg bg-gray-800 text-gray-100 border border-gray-600 focus:border-violet-500 focus:ring-1 focus:ring-violet-500 transition-colors outline-none disabled:opacity-50 disabled:cursor-not-allowed";

  return (
    <div
      id="setting-container"
      className="container bg-gray-800 rounded-2xl shadow-xl p-5 mb-6 border border-gray-700"
    >
      <div className="flex items-center justify-between mb-5 border-b border-gray-700 pb-3">
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          <span className="w-2 h-6 bg-violet-500 rounded-full inline-block"></span>
          Settings
        </h2>
        {activeFeature && (
          <span className="text-xs font-medium px-2 py-1 bg-violet-500/20 text-violet-300 rounded border border-violet-500/30 animate-pulse">
            Model Running
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Group 1: Model Configuration */}
        <div className={sectionClass}>
          <h3 className="text-gray-200 font-medium mb-4 flex items-center gap-2">
            Model Configuration
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="flex flex-col">
              <label className={labelClass}>Task</label>
              <select
                onChange={(e) => {
                  modelConfigRef.current.task = e.target.value;
                  loadModel();
                }}
                disabled={activeFeature !== null}
                className={inputClass}
              >
                <option value="detect">Object Detection</option>
                <option value="pose">Pose Estimation</option>
                <option value="seg">Segmentation</option>
              </select>
            </div>

            <div className="flex flex-col">
              <label className={labelClass}>Model</label>
              <select
                onChange={(e) => {
                  const selectedModel = e.target.value;
                  modelConfigRef.current.model = selectedModel;

                  if (
                    // yolo11 and yolo12 require NMS, lock it on
                    selectedModel.startsWith("yolo11") ||
                    selectedModel.startsWith("yolo12")
                  ) {
                    setNmsEnabled(true);
                    setNmsLocked(true);
                    modelConfigRef.current.enableNMS = true;
                  } else if (selectedModel.startsWith("yolo26")) {
                    // yolo26 has built-in NMS, lock it on
                    setNmsEnabled(false);
                    setNmsLocked(true);
                    modelConfigRef.current.enableNMS = false;
                  } else {
                    // Custom models, allow user to toggle NMS
                    setNmsEnabled(true);
                    setNmsLocked(false);
                    modelConfigRef.current.enableNMS = true;
                  }

                  loadModel();
                }}
                disabled={activeFeature !== null}
                className={inputClass}
              >
                <option value="yolo11n">YOLO11n (2.6M)</option>
                <option value="yolo11s">YOLO11s (9.4M)</option>
                <option value="yolo12n">YOLO12n (2.6M)</option>
                <option value="yolo12s">YOLO12s (9.3M)</option>
                <option value="yolo26n">YOLO26n (2.4M)</option>
                <option value="yolo26s">YOLO26s (9.5M)</option>
                {customModels.map((model, index) => (
                  <option key={index} value={model.url}>
                    {model.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex flex-col sm:col-span-2">
              <label className={labelClass}>Classes</label>
              <select
                defaultValue="default"
                disabled={activeFeature !== null}
                onChange={(e) => {
                  if (e.target.value === "default") {
                    modelConfigRef.current.classes = defaultClasses;
                  } else {
                    const selectedIndex = parseInt(e.target.value);
                    modelConfigRef.current.classes =
                      customClasses[selectedIndex].data;
                  }
                }}
                className={inputClass}
              >
                <option value="default">Default Classes (COCO)</option>
                {customClasses.map((classFile, index) => (
                  <option key={index} value={index}>
                    {classFile.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex flex-col sm:col-span-2">
              <label className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg border border-gray-600/50 hover:bg-gray-700/50 cursor-pointer transition-colors group">
                <div className="flex flex-col">
                  <span className="text-gray-200 font-medium text-sm group-hover:text-white transition-colors">
                    Non-Maximum Suppression (NMS)
                  </span>
                  <span className="text-gray-400 text-xs">
                    {nmsLocked
                      ? nmsEnabled
                        ? "Required & Locked for this model"
                        : "Disabled & Locked for YOLO26"
                      : "Optional for Custom Models"}
                  </span>
                </div>
                <div className="relative flex items-center">
                  <input
                    type="checkbox"
                    checked={nmsEnabled}
                    onChange={(e) => {
                      const isChecked = e.target.checked;
                      setNmsEnabled(isChecked);
                      modelConfigRef.current.enableNMS = isChecked;
                    }}
                    disabled={activeFeature !== null || nmsLocked}
                    className="peer sr-only "
                  />
                  <div
                    className={`w-11 h-6 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all 
                    ${activeFeature !== null || nmsLocked ? "opacity-50 cursor-not-allowed" : ""}
                    ${nmsEnabled ? "peer-checked:bg-violet-600 bg-violet-600" : "bg-gray-700"}
                  `}
                  ></div>
                </div>
              </label>
            </div>
          </div>
        </div>

        {/* Group 2: Execution Environment */}
        <div className={sectionClass}>
          <h3 className="text-gray-200 font-medium mb-4 flex items-center gap-2">
            Execution Environment
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="flex flex-col">
              <label className={labelClass}>Backend</label>
              <select
                onChange={(e) => {
                  modelConfigRef.current.backend = e.target.value;
                  loadModel();
                }}
                disabled={activeFeature !== null}
                className={inputClass}
              >
                <option value="wasm">Wasm (CPU)</option>
                <option value="webgpu">WebGPU</option>
              </select>
            </div>

            <div className="flex flex-col">
              <label className={labelClass}>Threads</label>
              <div className="relative">
                <input
                  type="number"
                  min="1"
                  max={navigator.hardwareConcurrency}
                  defaultValue={1}
                  disabled={activeFeature !== null}
                  onChange={handleThreadChange}
                  className={`${inputClass} pr-8`}
                  placeholder="4"
                />
                <div className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 text-xs pointer-events-none">
                  cores
                </div>
              </div>
            </div>

            <div className="flex flex-col">
              <label className={labelClass}>Camera Source</label>
              <select
                ref={cameraSelectorRef}
                disabled={activeFeature !== null}
                className={inputClass}
              >
                {cameras.length === 0 ? (
                  <option value="">No cameras detected</option>
                ) : (
                  cameras.map((camera, index) => (
                    <option key={index} value={camera.deviceId}>
                      {camera.label || `Camera ${index + 1}`}
                    </option>
                  ))
                )}
              </select>
            </div>

            <div className="flex flex-col">
              <label className={labelClass}>Image Strategy</label>
              <select
                disabled={activeFeature !== null}
                ref={imgszTypeSelectorRef}
                onChange={(e) => {
                  modelConfigRef.current.imgsz_type = e.target.value;
                }}
                className={inputClass}
              >
                <option value="dynamic">Dynamic (Resize)</option>
                <option value="zeroPad">Zero Padding</option>
              </select>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

export default SettingsPanel;

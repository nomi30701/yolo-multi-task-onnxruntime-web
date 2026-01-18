import React, { memo } from "react";

const ControlButtons = memo(function ControlButtons({
  imgSrc,
  fileVideoRef,
  fileImageRef,
  handle_OpenVideo,
  handle_OpenImage,
  handle_ToggleCamera,
  handle_AddModel,
  handle_AddClassesFile,
  activeFeature,
}) {
  const btnBase =
    "group flex items-center justify-center p-3 sm:p-4 rounded-xl font-bold transition-all duration-200 active:scale-95 shadow-lg border";

  const btnPrimary =
    "bg-gray-700/50 hover:bg-gray-700 text-violet-300 border-violet-500/30 hover:border-violet-500/60 hover:shadow-violet-500/10";

  const btnActive =
    "bg-red-500/10 hover:bg-red-500/20 text-red-400 border-red-500/30 hover:border-red-500/60";

  const btnSecondary =
    "bg-gray-700/30 hover:bg-gray-700 text-gray-300 border-gray-600/30 hover:border-gray-500 hover:text-white";

  return (
    <div className="container bg-gray-800 rounded-2xl shadow-xl p-5 mb-6 border border-gray-700">
      <div className="flex items-center justify-between mb-5 border-b border-gray-700 pb-3">
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          <span className="w-2 h-6 bg-violet-500 rounded-full inline-block"></span>
          Controls
        </h2>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Hidden Inputs */}
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

        {/* --- Primary Controls --- */}

        {/* Video Button (Hidden in logic but kept structure) */}
        <button
          className={`${btnBase} ${btnPrimary} hidden`}
          onClick={() => fileVideoRef.current.click()}
          disabled={activeFeature !== null}
        >
          <span className="truncate">Open Video</span>
        </button>

        {/* Image Control */}
        <button
          className={`${btnBase} ${
            activeFeature === "image" ? btnActive : btnPrimary
          }`}
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
                viewBox="0 0 24 24"
                stroke="currentColor"
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
                viewBox="0 0 24 24"
                stroke="currentColor"
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

        {/* Camera Control */}
        <button
          className={`${btnBase} ${
            activeFeature === "camera" ? btnActive : btnPrimary
          }`}
          onClick={handle_ToggleCamera}
          disabled={activeFeature !== null && activeFeature !== "camera"}
        >
          {activeFeature === "camera" ? (
            <>
              <svg
                className="w-5 h-5 mr-2"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
              Stop Camera
            </>
          ) : (
            <>
              <svg
                className="w-5 h-5 mr-2"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                />
              </svg>
              Open Camera
            </>
          )}
        </button>

        {/* --- Secondary Actions --- */}

        <button
          className={`${btnBase} ${btnSecondary}`}
          onClick={() => {
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
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 6v6m0 0v6m0-6h6m-6 0H6"
            />
          </svg>
          Add Model
        </button>

        <button
          className={`${btnBase} ${btnSecondary}`}
          onClick={() => {
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
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          Add Classes
        </button>
      </div>
    </div>
  );
});

export default ControlButtons;

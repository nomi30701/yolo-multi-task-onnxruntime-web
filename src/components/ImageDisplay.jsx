import React, { memo } from "react";

const ImageDisplay = memo(function ImageDisplay({
  cameraRef,
  imgRef,
  overlayRef,
  imgSrc,
  onCameraLoad,
  onImageLoad,
  activeFeature,
}) {
  return (
    <div className="container bg-gray-800 rounded-2xl shadow-xl border border-gray-700 p-5 mb-6 relative min-h-[400px] flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          <span className="w-2 h-6 bg-violet-500 rounded-full inline-block"></span>
          Preview
        </h2>

        {activeFeature && (
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-gray-700/50 border border-gray-600/50">
            <span
              className={`w-2 h-2 rounded-full ${activeFeature === "camera" ? "bg-red-500 animate-pulse" : "bg-violet-500"}`}
            ></span>
            <span className="text-xs font-semibold uppercase tracking-wider text-gray-300">
              {activeFeature === "camera" ? "Live Stream" : "Static Image"}
            </span>
          </div>
        )}
      </div>

      <div className="flex-1 rounded-xl bg-gray-900/50 border-2 border-dashed border-gray-700 relative overflow-hidden flex items-center justify-center">
        {activeFeature === null && (
          <div className="text-center p-8">
            <div className="w-20 h-20 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-4 border border-gray-700 shadow-inner">
              <svg
                className="w-10 h-10 text-gray-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
            </div>
            <h3 className="text-lg font-bold text-gray-300 mb-1">
              No Media Selected
            </h3>
            <p className="text-gray-500 text-sm max-w-xs mx-auto">
              Choose a source from the control panel below to start detection
            </p>
          </div>
        )}

        <div className="relative w-full h-full flex items-center justify-center" hidden={activeFeature === null}>
          <video
            className="max-h-[600px] w-full object-contain"
            ref={cameraRef}
            onLoadedMetadata={onCameraLoad}
            hidden={activeFeature !== "camera"}
            autoPlay
            playsInline
            muted
          />
          <img
            id="img"
            ref={imgRef}
            src={imgSrc}
            onLoad={onImageLoad}
            hidden={activeFeature !== "image"}
            className="max-h-[600px] w-full object-contain"
            alt="Source"
          />
          <canvas
            ref={overlayRef}
            hidden={activeFeature === null}
            className="absolute top-0 left-0 w-full h-full pointer-events-none"
            style={{ objectFit: "contain" }}
          ></canvas>
        </div>
      </div>
    </div>
  );
});

export default ImageDisplay;

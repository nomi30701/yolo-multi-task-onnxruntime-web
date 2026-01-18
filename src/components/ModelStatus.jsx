import React, { memo } from "react";

const ModelStatus = memo(function ModelStatus({
  warnUpTime,
  inferenceTime,
  statusMsg,
  statusColor,
}) {
  return (
    <div className="container bg-gray-800 rounded-2xl shadow-xl p-5 mb-6 border border-gray-700">
      <div className="flex items-center justify-between mb-5 border-b border-gray-700 pb-3">
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          <span className="w-2 h-6 bg-violet-500 rounded-full inline-block"></span>
          Performance Metrics
        </h2>
        {statusMsg && (
          <div
            className={`px-3 py-1 rounded-md text-xs font-mono border ${
              statusColor === "green"
                ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                : statusColor === "red"
                  ? "bg-red-500/10 text-red-400 border-red-500/20"
                  : statusColor === "blue"
                    ? "bg-blue-500/10 text-blue-400 border-blue-500/20"
                    : "bg-gray-700 text-gray-400 border-gray-600"
            }`}
          >
            STATUS: {statusMsg.toUpperCase()}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Warm Up Time Card */}
        <div className="bg-gray-700/30 rounded-xl p-4 border border-gray-600/30 flex items-center justify-between group hover:border-violet-500/30 transition-colors">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-orange-500/10 text-orange-400">
              <svg
                className="w-5 h-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 10V3L4 14h7v7l9-11h-7z"
                />
              </svg>
            </div>
            <div>
              <div className="text-gray-400 text-xs uppercase tracking-wider font-semibold">
                Warm Up Time
              </div>
              <div className="text-2xl font-bold text-white tracking-tight">
                {warnUpTime}{" "}
                <span className="text-sm font-normal text-gray-500">ms</span>
              </div>
            </div>
          </div>
        </div>

        {/* Inference Time Card */}
        <div className="bg-gray-700/30 rounded-xl p-4 border border-gray-600/30 flex items-center justify-between group hover:border-violet-500/30 transition-colors">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400">
              <svg
                className="w-5 h-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            </div>
            <div>
              <div className="text-gray-400 text-xs uppercase tracking-wider font-semibold">
                Inference Time
              </div>
              <div className="text-2xl font-bold text-white tracking-tight">
                {inferenceTime}{" "}
                <span className="text-sm font-normal text-gray-500">ms</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Visual Status Indicator */}
      <div className="mt-4 pt-4 border-t border-gray-700/50 flex items-center gap-2 text-sm text-gray-400">
        <div
          className={`w-2 h-2 rounded-full ${statusColor === "green" ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" : statusColor === "red" ? "bg-red-500" : "bg-blue-500 animate-pulse"}`}
        ></div>
        <span>
          System is{" "}
          {statusColor === "green"
            ? "ready"
            : statusColor === "blue"
              ? "processing"
              : "busy"}
        </span>
      </div>
    </div>
  );
});

export default ModelStatus;

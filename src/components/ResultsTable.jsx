import React, { memo } from "react";

const ResultsTable = memo(function ResultsTable({ details, currentClasses }) {
  return (
    <div className="container bg-gray-800 rounded-2xl shadow-xl p-5 mb-6 border border-gray-700">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          <span className="w-2 h-6 bg-violet-500 rounded-full inline-block"></span>
          Detections
          <span className="ml-2 text-sm font-medium bg-gray-700 text-gray-300 px-2 py-0.5 rounded-full border border-gray-600">
            {details.length}
          </span>
        </h2>
      </div>

      <div className="mt-4 overflow-hidden rounded-xl border border-gray-700 bg-gray-900/50">
        {details.length === 0 ? (
          <div className="p-8 text-center flex flex-col items-center">
            <div className="w-12 h-12 bg-gray-800 rounded-lg flex items-center justify-center mb-3">
              <svg
                className="w-6 h-6 text-gray-600"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                />
              </svg>
            </div>
            <p className="text-gray-500 font-medium">No objects detected yet</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-gray-800/80 border-b border-gray-700 text-xs uppercase tracking-wider text-gray-400 font-semibold">
                  <th className="p-4 w-20">ID</th>
                  <th className="p-4">Object Class</th>
                  <th className="p-4 w-48">Confidence Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-700/50">
                {details.map((item, index) => (
                  <tr
                    key={index}
                    className="hover:bg-violet-500/5 transition-colors group"
                  >
                    <td className="p-4 text-gray-500 font-mono text-sm group-hover:text-violet-400">
                      #{index}
                    </td>
                    <td className="p-4 font-medium text-white group-hover:text-violet-200">
                      {currentClasses[item.classIdx] ||
                        `Class ${item.classIdx}`}
                    </td>
                    <td className="p-4">
                      <div className="flex items-center gap-3">
                        <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-500"
                            style={{ width: `${item.score * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-sm font-mono text-gray-300 w-12 text-right">
                          {(item.score * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
});

export default ResultsTable;

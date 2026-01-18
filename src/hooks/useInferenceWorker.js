import { useRef, useEffect, useCallback } from "react";

export const useInferenceWorker = ({ onModelLoaded, onResult, onError }) => {
  const workerRef = useRef(null);

  // Keep callbacks fresh in ref to avoid re-creating worker on callback change
  const callbacksRef = useRef({ onModelLoaded, onResult, onError });
  useEffect(() => {
    callbacksRef.current = { onModelLoaded, onResult, onError };
  }, [onModelLoaded, onResult, onError]);

  useEffect(() => {
    const worker = new Worker(
      new URL("../utils/inference-pipeline-worker.js", import.meta.url),
      { type: "module" }
    );

    worker.onmessage = (e) => {
      const { type } = e.data;
      if (type === "MODEL_LOADED") {
        callbacksRef.current.onModelLoaded?.(e.data);
      } else if (type === "RESULT") {
        callbacksRef.current.onResult?.(e.data);
      } else if (type === "MODEL_LOAD_ERROR") {
        callbacksRef.current.onError?.(e.data);
      }
    };

    workerRef.current = worker;

    return () => {
      worker.terminate();
    };
  }, []);

  const postMessage = useCallback((message, transfer) => {
    workerRef.current?.postMessage(message, transfer);
  }, []);

  return { postMessage };
};

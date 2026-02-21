import { InferenceSession, Tensor, env } from "onnxruntime-web/webgpu";

export async function modelLoader(model_path, backend, numThreads = 1) {
  const DEFAULT_INPUT_SIZE = [1, 3, 640, 640];

  // activate SIMD (faster cpu), multi-threading, if wasm backend
  // default is false, 1 thread
  // env.wasm.simd = true;
  // env.wasm.numThreads = numThreads;

  let yolo_model;
  const sessionOptions = {
    executionProviders: [backend],
    // graphOptimizationLevel: "all",
    enableCpuMemArena: false, // true: more memory but faster
    enableMemPattern: false, // same.
  };

  try {
    // load model
    yolo_model = await InferenceSession.create(model_path, sessionOptions);
  } catch (e) {
    console.error("Failed to create inference session with SIMD:", e);
    if (backend === "wasm" && env.wasm.simd) {
      console.warn("Retrying without SIMD...");
      env.wasm.simd = false;
      yolo_model = await InferenceSession.create(model_path, sessionOptions);
    } else {
      throw e;
    }
  }

  // warm up
  const dummy_input_tensor = new Tensor(
    "float32",
    new Float32Array(DEFAULT_INPUT_SIZE.reduce((a, b) => a * b)),
    DEFAULT_INPUT_SIZE,
  );
  const warmupResults = await yolo_model.run({ images: dummy_input_tensor });
  for (const key in warmupResults) {
    warmupResults[key].dispose();
  }
  dummy_input_tensor.dispose();

  return yolo_model;
}

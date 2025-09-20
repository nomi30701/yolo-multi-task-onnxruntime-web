import * as ort from "onnxruntime-web/webgpu";

// Model loading function
async function loadModel(modelPath, backend) {
  self.postMessage({
    statusMsg: "Initializing model loader...",
    color: "blue",
  });

  // Load the model
  try {
    const yolo_model = await ort.InferenceSession.create(modelPath, {
      executionProviders: [backend],
    });

    console.log(yolo_model);

    self.postMessage({
      statusMsg: "Model loaded, preparing execution environment...",
      color: "blue",
    });

    return yolo_model;
  } catch (error) {
    throw new Error(`Model loading failed: ${error.message}`);
  }
}

// Model warmup function
async function warmupModel(model, inputShape) {
  try {
    self.postMessage({
      statusMsg: "Warming up model...",
      color: "blue",
    });

    const dummy_input_tensor = new ort.Tensor(
      "float32",
      new Float32Array(inputShape.reduce((a, b) => a * b)),
      inputShape
    );

    const { output0 } = await model.run({ images: dummy_input_tensor });
    output0.dispose();
    dummy_input_tensor.dispose();

    return true;
  } catch (error) {
    throw new Error(`Model warmup failed: ${error.message}`);
  }
}

// Worker message handler
self.onmessage = async function (e) {
  const start = performance.now();
  let yolo_model = null;

  try {
    const { model_path, backend, input_shape } = e.data;

    // load
    yolo_model = await loadModel(model_path, backend);

    // warmup
    await warmupModel(yolo_model, input_shape);

    const end = performance.now();
    const loadTime = end - start;

    // Send message
    self.postMessage({
      yolo_model,
      eps: loadTime,
      statusMsg: `Model loaded successfully in ${loadTime.toFixed(1)}ms`,
      color: "green",
    });
  } catch (error) {
    console.error("Model loader error:", error);

    self.postMessage({
      statusMsg: error.message || "Unknown model loading error",
      color: "red",
    });
  }
};

import * as ort from "onnxruntime-web/webgpu";

export async function model_loader(model_path, config) {
  ort.env.wasm.wasmPaths = `./`;

  // load model
  const yolo_model = await ort.InferenceSession.create(model_path, {
    executionProviders: [config.backend],
  });

  // warm up
  const dummy_input_tensor = new ort.Tensor(
    "float32",
    new Float32Array(config.input_shape.reduce((a, b) => a * b)),
    config.input_shape
  );
  const { output0 } = await yolo_model.run({ images: dummy_input_tensor });
  output0.dispose();
  dummy_input_tensor.dispose();
  
  return yolo_model;
}

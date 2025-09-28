import { InferenceSession, Tensor } from "onnxruntime-web/webgpu";

export async function model_loader(model_path, backend) {
  const DEFAULT_INPUT_SIZE = [1, 3, 640, 640];

  // load model
  const yolo_model = await InferenceSession.create(model_path, {
    executionProviders: [backend],
  });

  // warm up
  const dummy_input_tensor = new Tensor(
    "float32",
    new Float32Array(DEFAULT_INPUT_SIZE.reduce((a, b) => a * b)),
    DEFAULT_INPUT_SIZE
  );
  const { output0 } = await yolo_model.run({ images: dummy_input_tensor });
  output0.dispose();
  dummy_input_tensor.dispose();

  return yolo_model;
}

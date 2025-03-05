import * as ort from "onnxruntime-web/webgpu";
import { preProcess_dynamic, applyNMS } from "./img_preprocess";

export const inference_pipeline = async (input_el, session, config) => {
  let input_tensor = null;
  let output0 = null;

  try {
    const src_mat = cv.imread(input_el);

    // const [src_mat_preProcessed, xRatio, yRatio] = await preProcess(
    //   src_mat,
    //   sessionsConfig.input_shape[2],
    //   sessionsConfig.input_shape[3]
    // );

    const [src_mat_preProcessed, div_width, div_height] =
      preProcess_dynamic(src_mat);
    const xRatio = src_mat.cols / div_width;
    const yRatio = src_mat.rows / div_height;

    src_mat.delete();

    input_tensor = new ort.Tensor("float32", src_mat_preProcessed.data32F, [
      1,
      3,
      div_height,
      div_width,
    ]);
    src_mat_preProcessed.delete();

    const start = performance.now();
    const { output0 } = await session.run({
      images: input_tensor,
    });
    const end = performance.now();
    input_tensor.dispose();

    // post process
    let results;
    switch (config.task) {
      case "detect":
        results = post_process_detect(output0, config, xRatio, yRatio);
        break;
      case "pose":
        results = post_process_pose(output0, config, xRatio, yRatio);
        break;
      case "segment":
        results = post_process_segment(output0, config, xRatio, yRatio);
        break;
      default:
        console.warn(`Unknown task: ${config.task}, falling back to detection`);
    }
    output0.dispose();

    // nms
    const selected_indices = applyNMS(
      results,
      results.map((r) => r.score),
      config.iou_threshold
    );
    const filtered_results = selected_indices.map((i) => results[i]);

    return [filtered_results, (end - start).toFixed(2)];
  } catch (error) {
    console.error("Inference error:", error);
    return [[], "0.00"];
  } finally {
    if (input_tensor) input_tensor.dispose();
    if (output0) output0.dispose();
  }
};

function post_process_detect(raw_tensor, config, xRatio, yRatio) {
  const NUM_PREDICTIONS = raw_tensor.dims[2];
  const NUM_BBOX_ATTRS = 4;
  const NUM_SCORES = 80;

  const predictions = raw_tensor.data;
  const bbox_data = predictions.subarray(0, NUM_PREDICTIONS * NUM_BBOX_ATTRS);
  const scores_data = predictions.subarray(NUM_PREDICTIONS * NUM_BBOX_ATTRS);

  const results = new Array(Math.min(50, NUM_PREDICTIONS));
  let resultCount = 0;

  for (let i = 0; i < NUM_PREDICTIONS; i++) {
    let maxScore = 0;
    let class_idx = -1;

    for (let c = 0; c < NUM_SCORES; c++) {
      const score = scores_data[i + c * NUM_PREDICTIONS];
      if (score > maxScore) {
        maxScore = score;
        class_idx = c;
      }
    }
    if (maxScore <= config.score_threshold) continue;

    const w = bbox_data[i + NUM_PREDICTIONS * 2] * xRatio;
    const h = bbox_data[i + NUM_PREDICTIONS * 3] * yRatio;
    const x = bbox_data[i] * xRatio - 0.5 * w;
    const y = bbox_data[i + NUM_PREDICTIONS] * yRatio - 0.5 * h;

    results[resultCount++] = {
      bbox: [x, y, w, h],
      class_idx,
      score: maxScore,
    };
  }
  return resultCount < results.length ? results.slice(0, resultCount) : results;
}

function post_process_pose(raw_tensor, config, xRatio, yRatio) {
  // post process
  const NUM_PREDICTIONS = raw_tensor.dims[2];
  const NUM_BBOX_ATTRS = 5;
  const NUM_KEYPOINTS = 17;
  const KEYPOINT_DIMS = 3;

  const predictions = raw_tensor.data;
  const bbox_data = predictions.subarray(0, NUM_PREDICTIONS * NUM_BBOX_ATTRS);
  const keypoints_data = predictions.subarray(NUM_PREDICTIONS * NUM_BBOX_ATTRS);

  const results = new Array(Math.min(50, NUM_PREDICTIONS));
  let resultCount = 0;

  for (let i = 0; i < NUM_PREDICTIONS; i++) {
    const score = bbox_data[i + NUM_PREDICTIONS * 4];
    if (score <= config.score_threshold) continue;

    const w = bbox_data[i + NUM_PREDICTIONS * 2] * xRatio;
    const h = bbox_data[i + NUM_PREDICTIONS * 3] * yRatio;
    const x = bbox_data[i] * xRatio - 0.5 * w;
    const y = bbox_data[i + NUM_PREDICTIONS] * yRatio - 0.5 * h;

    const keypoints = new Array(NUM_KEYPOINTS);
    for (let kp = 0; kp < NUM_KEYPOINTS; kp++) {
      const base_idx = kp * KEYPOINT_DIMS * NUM_PREDICTIONS + i;
      keypoints[kp] = {
        x: keypoints_data[base_idx] * xRatio,
        y: keypoints_data[base_idx + NUM_PREDICTIONS] * yRatio,
        score: keypoints_data[base_idx + NUM_PREDICTIONS * 2],
      };
    }

    results[resultCount++] = {
      bbox: [x, y, w, h],
      score,
      keypoints,
    };
  }
  return resultCount < results.length ? results.slice(0, resultCount) : results;
}
function post_process_segment() {
  return;
}

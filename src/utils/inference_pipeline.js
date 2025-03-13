import * as ort from "onnxruntime-web/webgpu";
import {
  preProcess_dynamic,
  preProcess,
  applyNMS,
  Colors,
} from "./img_preprocess";

export const inference_pipeline = async (
  input_el,
  session,
  config,
  overlay_el
) => {
  let input_tensor = null;
  let output0 = null;
  let output1 = null;

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
    const outputs = await session.run({
      images: input_tensor,
    });
    const end = performance.now();
    input_tensor.dispose();
    output0 = outputs.output0;
    output1 = outputs.output1;

    // post process
    let results;
    let masksData;
    switch (config.task) {
      case "detect":
        results = post_process_detect(output0, config, xRatio, yRatio);
        break;
      case "pose":
        results = post_process_pose(output0, config, xRatio, yRatio);
        break;
      case "segment":
        [results, masksData] = post_process_segment(
          output0,
          output1,
          config,
          xRatio,
          yRatio
        );
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

    if (config.task === "segment" && filtered_results.length > 0) {
      renderSegmentationMasks(filtered_results, masksData, overlay_el);
    }

    return [filtered_results, (end - start).toFixed(2)];
  } catch (error) {
    console.error("Inference error:", error);
    return [[], "0.00"];
  } finally {
    if (input_tensor) input_tensor.dispose();
    if (output0) output0.dispose();
    if (output1) output1.dispose();
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

function post_process_segment(output0, output1, config, xRatio, yRatio) {
  const NUM_PREDICTIONS = output0.dims[2];
  const NUM_BBOX_ATTRS = 4;
  const NUM_SCORES = 80;
  const NUM_MASK_WEIGHTS = 32;

  const predictions = output0.data;
  const bbox_data = predictions.subarray(0, NUM_PREDICTIONS * NUM_BBOX_ATTRS);
  const scores_data = predictions.subarray(
    NUM_PREDICTIONS * NUM_BBOX_ATTRS,
    NUM_PREDICTIONS * (NUM_BBOX_ATTRS + NUM_SCORES)
  );
  const mask_weights_data = predictions.subarray(
    NUM_PREDICTIONS * (NUM_BBOX_ATTRS + NUM_SCORES)
  );

  const proto_mask = output1.data;
  const MASK_CHANNELS = output1.dims[1];
  const MASK_HEIGHT = output1.dims[2];
  const MASK_WIDTH = output1.dims[3];

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

    const mask_weights = new Float32Array(NUM_MASK_WEIGHTS);
    for (let c = 0; c < NUM_MASK_WEIGHTS; c++) {
      mask_weights[c] = mask_weights_data[i + c * NUM_PREDICTIONS];
    }

    results[resultCount++] = {
      bbox: [x, y, w, h],
      class_idx,
      score: maxScore,
      mask_weights,
    };
  }

  const masksData = {
    proto_mask,
    MASK_CHANNELS,
    MASK_HEIGHT,
    MASK_WIDTH,
  };

  const finalResults =
    resultCount < results.length ? results.slice(0, resultCount) : results;

  return [finalResults, masksData];
}

function renderSegmentationMasks(filtered_results, masksData, overlay_el) {
  const { proto_mask, MASK_CHANNELS, MASK_HEIGHT, MASK_WIDTH } = masksData;

  const proto_mask_mat = cv.matFromArray(
    MASK_CHANNELS,
    MASK_HEIGHT * MASK_WIDTH,
    cv.CV_32F,
    proto_mask
  );

  try {
    // weights x proto_mask
    const NUM_FILTERED_RESULTS = filtered_results.length;
    const mask_weights = filtered_results
      .map((r) => Array.from(r.mask_weights))
      .flat();
    const mask_weights_mat = cv.matFromArray(
      NUM_FILTERED_RESULTS,
      MASK_CHANNELS,
      cv.CV_32F,
      mask_weights
    );
    const weights_mul_proto_mat = new cv.Mat();

    const temp_mat1 = new cv.Mat();
    cv.gemm(
      mask_weights_mat,
      proto_mask_mat,
      1.0,
      temp_mat1,
      0.0,
      weights_mul_proto_mat
    );
    temp_mat1.delete();

    proto_mask_mat.delete();
    mask_weights_mat.delete();

    const mask_sigmoid_mat = new cv.Mat();
    const ones_mat = cv.Mat.ones(weights_mul_proto_mat.size(), cv.CV_32F);

    const temp_mat2 = new cv.Mat(
      weights_mul_proto_mat.rows,
      weights_mul_proto_mat.cols,
      cv.CV_32F,
      new cv.Scalar(-1)
    );
    cv.multiply(weights_mul_proto_mat, temp_mat2, mask_sigmoid_mat);
    temp_mat2.delete();

    cv.exp(mask_sigmoid_mat, mask_sigmoid_mat);
    cv.add(mask_sigmoid_mat, ones_mat, mask_sigmoid_mat);
    cv.divide(ones_mat, mask_sigmoid_mat, mask_sigmoid_mat);

    ones_mat.delete();
    weights_mul_proto_mat.delete();

    const overlay_mat = new cv.Mat(
      overlay_el.height,
      overlay_el.width,
      cv.CV_8UC4,
      new cv.Scalar(0, 0, 0, 0)
    );

    for (let i = 0; i < NUM_FILTERED_RESULTS; i++) {
      const mask = mask_sigmoid_mat.row(i).data32F;
      const mask_mat = cv.matFromArray(
        MASK_HEIGHT,
        MASK_WIDTH,
        cv.CV_32F,
        mask
      );

      const mask_resized_mat = new cv.Mat();
      cv.resize(
        mask_mat,
        mask_resized_mat,
        new cv.Size(overlay_el.width, overlay_el.height),
        cv.INTER_LINEAR
      );

      const mask_binary_mat = new cv.Mat();
      const mask_binary_u8_mat = new cv.Mat();
      cv.threshold(mask_resized_mat, mask_binary_mat, 0.5, 1, cv.THRESH_BINARY);
      mask_binary_mat.convertTo(mask_binary_u8_mat, cv.CV_8U);

      const [x, y, w, h] = filtered_results[i].bbox;
      const x1 = Math.max(0, x);
      const y1 = Math.max(0, y);
      const x2 = Math.min(overlay_el.width, x + w);
      const y2 = Math.min(overlay_el.height, y + h);

      const roi = mask_binary_u8_mat.roi(new cv.Rect(x1, y1, x2 - x1, y2 - y1));
      const color = Colors.getColor(filtered_results[i].class_idx, 0.6);
      const color_scalar = new cv.Scalar(
        color[0],
        color[1],
        color[2],
        color[3] * 255
      );
      const mask_colored_mat = new cv.Mat(
        roi.rows,
        roi.cols,
        cv.CV_8UC4,
        color_scalar
      );
      mask_colored_mat.copyTo(
        overlay_mat.roi(new cv.Rect(x1, y1, x2 - x1, y2 - y1)),
        roi
      );

      // release mat
      mask_colored_mat.delete();
      roi.delete();
      mask_binary_u8_mat.delete();
      mask_binary_mat.delete();
      mask_resized_mat.delete();
      mask_mat.delete();
    }
    mask_sigmoid_mat.delete();

    const imgData = new ImageData(
      new Uint8ClampedArray(overlay_mat.data),
      overlay_el.width,
      overlay_el.height
    );
    const ctx = overlay_el.getContext("2d");
    ctx.clearRect(0, 0, overlay_el.width, overlay_el.height);
    ctx.putImageData(imgData, 0, 0);

    overlay_mat.delete();
  } catch (error) {
    console.error("Error rendering masks:", error);
    proto_mask_mat.delete();
  }
}

import * as ort from "onnxruntime-web/webgpu";
import cv from "@techstark/opencv-js";
import { preProcess_img, applyNMS, Colors } from "./img_preprocess";

/**
 * Inference pipeline for YOLO model.
 * @param {cv.Mat} src_mat - Input image Mat.
 * @param {ort.InferenceSession} session - YOLO model session.
 * @param {[Number, Number]} overlay_size - Overlay width and height. [width, height]
 * @param {object} model_config - Model configuration object.
 * @returns {[Array[Object], Number]} - Array of predictions and inference time.
 */
export async function inference_pipeline(
  src_mat,
  session,
  overlay_size,
  model_config
) {
  // Pre-process img, inference
  let input_tensor, output0, output1;
  try {
    const [input_tensor, xRatio, yRatio] = preProcess_img(
      src_mat,
      overlay_size,
      model_config.imgsz_type
    );
    src_mat.delete();

    const start = performance.now();
    const outputs = await session.run({
      images: input_tensor,
    });
    const end = performance.now();
    input_tensor.dispose();
    output0 = outputs.output0;
    output1 = outputs.output1;

    // Post process
    let results, masksData, mask_imgData;
    switch (model_config.task) {
      case "detect":
        results = postProcess_detect(
          output0,
          model_config.iou_threshold,
          xRatio,
          yRatio
        );
        break;
      case "pose":
        results = postProcess_pose(
          output0,
          model_config.iou_threshold,
          xRatio,
          yRatio
        );
        break;
      case "segment":
        [results, masksData] = postProcess_segment(
          output0,
          output1,
          model_config.iou_threshold,
          xRatio,
          yRatio
        );
        break;
      default:
        console.warn(
          `Unknown task: ${model_config.task}, falling back to detection`
        );
    }
    output0.dispose();

    // nms
    const selected_indices = applyNMS(
      results,
      results.map((r) => r.score),
      model_config.iou_threshold
    );
    const filtered_results = selected_indices.map((i) => results[i]);

    if (model_config.task === "segment") {
      mask_imgData = postProcess_mask(
        filtered_results,
        masksData,
        overlay_size
      );
    }

    return [
      { bbox_results: filtered_results, mask_imgData },
      (end - start).toFixed(2),
    ];
  } catch (error) {
    console.error("Inference error:", error);
    return [[], "0.00"];
  } finally {
    if (input_tensor) input_tensor.dispose();
    if (output0) output0.dispose();
    if (output1) output1.dispose();
  }
}

/**
 *
 * @param {*} raw_tensor - yolo model output0
 * @param {*} score_threshold - score threshold
 * @param {*} xRatio - xRatio
 * @param {*} yRatio - yRatio
 * @returns - object detection results
 */
function postProcess_detect(
  raw_tensor,
  score_threshold = 0.45,
  xRatio,
  yRatio
) {
  const NUM_PREDICTIONS = raw_tensor.dims[2];
  const NUM_BBOX_ATTRS = 4;
  const NUM_SCORES = 80;

  const predictions = raw_tensor.data;
  const bbox_data = predictions.subarray(0, NUM_PREDICTIONS * NUM_BBOX_ATTRS);
  const scores_data = predictions.subarray(NUM_PREDICTIONS * NUM_BBOX_ATTRS);

  const results = new Array();
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
    if (maxScore <= score_threshold) continue;

    const w = bbox_data[i + NUM_PREDICTIONS * 2] * xRatio;
    const h = bbox_data[i + NUM_PREDICTIONS * 3] * yRatio;
    const tlx = bbox_data[i] * xRatio - 0.5 * w;
    const tly = bbox_data[i + NUM_PREDICTIONS] * yRatio - 0.5 * h;

    results[resultCount++] = {
      bbox: [tlx, tly, w, h],
      class_idx,
      score: maxScore,
    };
  }
  return results;
}

/**
 *
 * @param {*} raw_tensor - yolo model output0
 * @param {*} score_threshold - score threshold
 * @param {*} xRatio - xRatio
 * @param {*} yRatio - yRatio
 * @returns - pose estimation results
 */
function postProcess_pose(raw_tensor, score_threshold = 0.45, xRatio, yRatio) {
  // post process
  const NUM_PREDICTIONS = raw_tensor.dims[2];
  const NUM_BBOX_ATTRS = 5;
  const NUM_KEYPOINTS = 17;
  const KEYPOINT_DIMS = 3;

  const predictions = raw_tensor.data;
  const bbox_data = predictions.subarray(0, NUM_PREDICTIONS * NUM_BBOX_ATTRS);
  const keypoints_data = predictions.subarray(NUM_PREDICTIONS * NUM_BBOX_ATTRS);

  const results = new Array();
  let resultCount = 0;

  for (let i = 0; i < NUM_PREDICTIONS; i++) {
    const score = bbox_data[i + NUM_PREDICTIONS * 4];
    if (score <= score_threshold) continue;

    const w = bbox_data[i + NUM_PREDICTIONS * 2] * xRatio;
    const h = bbox_data[i + NUM_PREDICTIONS * 3] * yRatio;
    const tlx = bbox_data[i] * xRatio - 0.5 * w;
    const tly = bbox_data[i + NUM_PREDICTIONS] * yRatio - 0.5 * h;

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
      bbox: [tlx, tly, w, h],
      score,
      keypoints,
    };
  }
  return results;
}

/**
 *
 * @param {*} output0 - yolo model output0
 * @param {*} output1 - yolo model output1
 * @param {*} score_threshold - score threshold
 * @param {*} xRatio - xRatio
 * @param {*} yRatio - yRatio
 * @returns
 */
function postProcess_segment(
  output0,
  output1,
  score_threshold,
  xRatio,
  yRatio
) {
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

  const results = new Array();
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
    if (maxScore <= score_threshold) continue;

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

  return [results, masksData];
}

/**
 *
 * @param {*} filtered_results - NMS filtered results
 * @param {*} masksData - output1 data (mask weights)
 * @param {*} overlay_size - Size of the overlay. [width, height]
 * @returns {ImageData} - ImageData object for the overlay
 */
function postProcess_mask(filtered_results, masksData, overlay_size) {
  if (!filtered_results || filtered_results.length === 0) return null;
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
      overlay_size[1],
      overlay_size[0],
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
        new cv.Size(overlay_size[0], overlay_size[1]),
        cv.INTER_LINEAR
      );

      const mask_binary_mat = new cv.Mat();
      const mask_binary_u8_mat = new cv.Mat();
      cv.threshold(mask_resized_mat, mask_binary_mat, 0.5, 1, cv.THRESH_BINARY);
      mask_binary_mat.convertTo(mask_binary_u8_mat, cv.CV_8U);

      const [x, y, w, h] = filtered_results[i].bbox;
      const x1 = Math.max(0, x);
      const y1 = Math.max(0, y);
      const x2 = Math.min(overlay_size[0], x + w);
      const y2 = Math.min(overlay_size[1], y + h);

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
      overlay_size[0],
      overlay_size[1]
    );
    overlay_mat.delete();

    return imgData;
  } catch (error) {
    console.error("Error masks:", error);
    proto_mask_mat.delete();
  }
}

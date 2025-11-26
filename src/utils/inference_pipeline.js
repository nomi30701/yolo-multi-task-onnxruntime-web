import cv from "@techstark/opencv-js";
import { preProcess_img, applyNMS, Colors } from "./img_preprocess";

/**
 * Inference pipeline for YOLO model.
 * @param {HTMLImageElement|HTMLCanvasElement|OffscreenCanvas} imageSource - Input image source
 * @param {ort.InferenceSession} session - YOLO model ort session.
 * @param {[Number, Number]} overlay_size - Overlay width and height. [width, height]
 * @param {object} model_config - Model configuration object.
 * @returns {[object, string]} Tuple containing:
 *   - First element: object with inference results:
 *     - bbox_results: Array<Object> - Filtered detection results after NMS, each containing:
 *       - bbox: [x, y, width, height] in original image coordinates
 *       - class_idx: Predicted class index
 *       - score: Confidence score (0-1)
 *       - keypoints?:  For pose tasks: [{x, y, score}] for each keypoint
 *       - mask_weights?: For segmentation tasks: mask coefficients
 *     - mask_imgData?: For segmentation tasks: RGBA overlay image with colored masks
 *   - Second element: Inference time in milliseconds (formatted to 2 decimal places)
 *
 */
export async function inference_pipeline(
  imageSource,
  session,
  overlay_size,
  model_config
) {
  try {
    // Read DOM to cv.Mat
    const src_mat = cv.imread(imageSource);

    // Pre-process img, inference
    const [input_tensor, xRatio, yRatio] = preProcess_img(
      src_mat,
      overlay_size,
      model_config.imgsz_type
    );
    src_mat.delete();

    const start = performance.now();
    const { output0, output1 } = await session.run({
      images: input_tensor,
    });
    const end = performance.now();
    input_tensor.dispose();

    // Post process
    let results, masksData, mask_imgData;
    switch (model_config.task) {
      case "detect":
        results = postProcess_detect(
          output0,
          model_config.score_threshold,
          xRatio,
          yRatio
        );
        break;
      case "pose":
        results = postProcess_pose(
          output0,
          model_config.score_threshold,
          xRatio,
          yRatio
        );
        break;
      case "segment":
        [results, masksData] = postProcess_segment(
          output0,
          output1,
          model_config.score_threshold,
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
    output1?.dispose();

    // Apply NMS
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
  }
}

/**
 * Post process detection raw outputs.
 *
 * @param {ort.Tensor} raw_tensor - Yolo model output0
 * @param {number} score_threshold - Score threshold
 * @param {number} xRatio - xRatio
 * @param {number} yRatio - yRatio
 * @returns {Array<Object>} Array of object detection results. Each item:
 * - bbox: [number, number, number, number]
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
 * @param {ort.Tensor} raw_tensor - Yolo model output0
 * @param {number} score_threshold - Score threshold
 * @param {number} xRatio - xRatio
 * @param {number} yRatio - yRatio
 * @returns {Array<Object>} Array of pose estimation results.
 * - [{bbox: [x, y, w, h], score, keypoints: [{x, y, score}]}, ...]
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
 * Post process segmentation raw outputs
 *
 * @param {ort.Tensor} output0 - YOLO model detection output (shape: [1, G, 4 + C + M])
 * @param {ort.Tensor} output1 - YOLO model prototype masks (shape: [1, M, Hm, Wm])
 * @param {number} score_threshold - Score threshold for filtering detections (0-1)
 * @param {number} xRatio - Horizontal scale ratio to map boxes to original image
 * @param {number} yRatio - Vertical scale ratio to map boxes to original image
 * @returns {[Array<Object>, Object]} Returns a tuple [results, masksData]
 *   - results: Array of instance results. Each item:
 *     {
 *       bbox: [number, number, number, number], // [x, y, w, h] in original image coords
 *       class_idx: number,                      // predicted class index
 *       score: number,                          // confidence score
 *       mask_weight_idx: number
 *     }
 *   - masksData: Object containing mask prototype info:
 *     {
 *       proto_mask: Float32Array,
 *       mask_weights_data: Float32Array,
 *       MASK_CHANNELS: number,
 *       MASK_HEIGHT: number,
 *       MASK_WIDTH: number
 *     }
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
    const tlx = bbox_data[i] * xRatio - 0.5 * w;
    const tly = bbox_data[i + NUM_PREDICTIONS] * yRatio - 0.5 * h;

    // OPTIMIZATION: Do not create new Float32Array here.
    // Just store the index (i) to extract weights later after NMS.
    results[resultCount++] = {
      bbox: [tlx, tly, w, h],
      class_idx,
      score: maxScore,
      mask_weight_idx: i,
    };
  }

  const masksData = {
    proto_mask,
    // Use slice() to create a safe copy, ensuring data persists after output0.dispose()
    mask_weights_data: mask_weights_data.slice(),
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
  const {
    proto_mask,
    mask_weights_data,
    MASK_CHANNELS,
    MASK_HEIGHT,
    MASK_WIDTH,
  } = masksData;

  // proto_mask: [1, 32*160*160] -> cv.Mat(32, 160*160)
  const proto_mask_mat = cv.matFromArray(
    MASK_CHANNELS,
    MASK_HEIGHT * MASK_WIDTH,
    cv.CV_32F,
    proto_mask
  );

  try {
    // Weights x Proto_mask
    const NUM_FILTERED_RESULTS = filtered_results.length;

    const NUM_PREDICTIONS = mask_weights_data.length / MASK_CHANNELS;
    const mask_weights = new Float32Array(NUM_FILTERED_RESULTS * MASK_CHANNELS);

    for (let i = 0; i < NUM_FILTERED_RESULTS; i++) {
      const base_idx = filtered_results[i].mask_weight_idx;
      for (let c = 0; c < MASK_CHANNELS; c++) {
        mask_weights[i * MASK_CHANNELS + c] =
          mask_weights_data[base_idx + c * NUM_PREDICTIONS];
      }
    }

    const mask_weights_mat = cv.matFromArray(
      NUM_FILTERED_RESULTS,
      MASK_CHANNELS,
      cv.CV_32F,
      mask_weights
    );

    const weights_mul_proto_mat = new cv.Mat();
    cv.gemm(
      mask_weights_mat, // [N, 32]
      proto_mask_mat, // [32, 160*160]
      1.0,
      new cv.Mat(),
      0.0,
      weights_mul_proto_mat, // [N, 160*160]
      0
    );

    proto_mask_mat.delete();
    mask_weights_mat.delete();

    // Sigmoid
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

    // Create mask overlay
    const overlay_mat = new cv.Mat(
      overlay_size[1],
      overlay_size[0],
      cv.CV_8UC4,
      new cv.Scalar(0, 0, 0, 0)
    );

    const mask_resized_mat = new cv.Mat();
    const mask_binary_mat = new cv.Mat();
    const mask_binary_u8_mat = new cv.Mat();

    for (let i = 0; i < NUM_FILTERED_RESULTS; i++) {
      const mask = mask_sigmoid_mat.row(i).data32F;
      const mask_mat = cv.matFromArray(
        MASK_HEIGHT,
        MASK_WIDTH,
        cv.CV_32F,
        mask
      );

      const [x, y, w, h] = filtered_results[i].bbox;

      // 1. Calculate coordinates on the 160x160 mask
      const scaleX = MASK_WIDTH / overlay_size[0];
      const scaleY = MASK_HEIGHT / overlay_size[1];

      const mask_x = Math.floor(Math.max(0, x * scaleX));
      const mask_y = Math.floor(Math.max(0, y * scaleY));
      const mask_w = Math.ceil(Math.min(MASK_WIDTH - mask_x, w * scaleX));
      const mask_h = Math.ceil(Math.min(MASK_HEIGHT - mask_y, h * scaleY));

      // Boundary check
      if (mask_w > 0 && mask_h > 0) {
        // 2. Crop the small region from 160x160 mask
        const mask_roi = mask_mat.roi(
          new cv.Rect(mask_x, mask_y, mask_w, mask_h)
        );

        // 3. Resize only this small region to the target bbox size
        const target_x = Math.max(0, Math.floor(x));
        const target_y = Math.max(0, Math.floor(y));
        const target_w = Math.min(overlay_size[0] - target_x, Math.ceil(w));
        const target_h = Math.min(overlay_size[1] - target_y, Math.ceil(h));

        if (target_w > 0 && target_h > 0) {
          cv.resize(
            mask_roi,
            mask_resized_mat,
            new cv.Size(target_w, target_h),
            cv.INTER_LINEAR
          );

          // Binarize
          cv.threshold(
            mask_resized_mat,
            mask_binary_mat,
            0.5,
            255,
            cv.THRESH_BINARY
          );
          mask_binary_mat.convertTo(mask_binary_u8_mat, cv.CV_8U);

          // Colorize mask
          const color = Colors.getColor(filtered_results[i].class_idx, 0.6);
          const color_scalar = new cv.Scalar(
            color[0],
            color[1],
            color[2],
            color[3] * 255
          );

          // Create colored mat with target size
          const mask_colored_mat = new cv.Mat(
            target_h,
            target_w,
            cv.CV_8UC4,
            color_scalar
          );

          // Copy to overlay mat at the specific bbox location
          mask_colored_mat.copyTo(
            overlay_mat.roi(
              new cv.Rect(target_x, target_y, target_w, target_h)
            ),
            mask_binary_u8_mat
          );

          mask_colored_mat.delete();
        }
        mask_roi.delete();
      }
      mask_mat.delete();
    }
    mask_resized_mat.delete();
    mask_binary_mat.delete();
    mask_binary_u8_mat.delete();
    mask_sigmoid_mat.delete();

    const imgData = new ImageData(
      new Uint8ClampedArray(
        overlay_mat.data.buffer,
        overlay_mat.data.byteOffset,
        overlay_mat.data.byteLength
      ),
      overlay_size[0],
      overlay_size[1]
    );
    overlay_mat.delete();

    return imgData;
  } catch (error) {
    console.error("Error masks:", error);
    proto_mask_mat.delete();
    return null;
  }
}

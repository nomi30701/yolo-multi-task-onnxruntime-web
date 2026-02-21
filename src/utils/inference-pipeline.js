import cv from "@techstark/opencv-js";
import { preProcessImage, Colors, applyNMS } from "./process-util";

/**
 * Inference pipeline for YOLO model.
 * @param {ImageData} imageData - Input image data.
 * @param {ort.InferenceSession} session - YOLO model ONNX Runtime session.
 * @param {object} config - Model configuration object.
 * @returns {Promise<object>} Inference result object containing detection results, mask image data, and inference time.
 */
export async function inferencePipeline(imageData, session, config) {
  const matsToDelete = [];
  let inputTensor = null;
  let outputs = null;
  let output0 = null;
  let output1 = null;

  try {
    // Convert ImageData to cv.Mat
    const srcMat = new cv.Mat(imageData.height, imageData.width, cv.CV_8UC4);
    srcMat.data.set(imageData.data);
    matsToDelete.push(srcMat);

    // Pre-process img, inference
    let xRatio, yRatio;
    [inputTensor, xRatio, yRatio] = preProcessImage(
      srcMat,
      config.overlaySize,
      config.imgszType,
    );

    const start = performance.now();
    outputs = await session.run({
      images: inputTensor,
    });

    // Get outputs by name if output0/output1 are not explicitly defined
    const outputNames = session.outputNames;
    output0 = outputs[outputNames[0]];
    if (outputNames.length > 1) {
      output1 = outputs[outputNames[1]];
    }
    const end = performance.now();

    // Post process
    let results, masksData, maskImgData;
    switch (config.task) {
      case "detect":
        if (config.enableNMS) {
          results = postProcessDetect(
            output0,
            config.scoreThreshold,
            xRatio,
            yRatio,
          );
        } else {
          results = postProcessDetectEnd2End(
            output0,
            config.scoreThreshold,
            xRatio,
            yRatio,
          );
        }
        break;
      case "pose":
        if (config.enableNMS) {
          results = postProcessPose(
            output0,
            config.scoreThreshold,
            xRatio,
            yRatio,
          );
        } else {
          results = postProcessPoseEnd2End(
            output0,
            config.scoreThreshold,
            xRatio,
            yRatio,
          );
        }
        break;
      case "seg":
        if (config.enableNMS) {
          [results, masksData] = postProcessSegment(
            output0,
            output1,
            config.scoreThreshold,
            xRatio,
            yRatio,
          );
        } else {
          [results, masksData] = postProcessSegmentEnd2End(
            output0,
            output1,
            config.scoreThreshold,
            xRatio,
            yRatio,
          );
        }
        break;
      default:
        console.warn(`Unknown task: ${config.task}, falling back to detection`);
    }

    // Apply NMS
    let filteredResults;
    if (config.enableNMS) {
      const selectedIndices = applyNMS(
        results,
        results.map((r) => r.score),
        config.iouThreshold,
      );

      filteredResults = selectedIndices.map((i) => results[i]);
    } else {
      filteredResults = results;
    }

    if (config.task === "seg") {
      maskImgData = postProcessMask(
        filteredResults,
        masksData,
        config.overlaySize,
      );
    }

    return {
      results: filteredResults,
      maskImageData: maskImgData,
      inferenceTime: (end - start).toFixed(2),
    };
  } catch (error) {
    console.error("Inference error:", error);
    return {
      results: [],
      maskImageData: null,
      inferenceTime: "0.00",
    };
  } finally {
    matsToDelete.forEach((mat) => {
      if (mat && !mat.isDeleted()) mat.delete();
    });
    if (inputTensor) inputTensor.dispose();
    if (outputs) {
      for (const key in outputs) {
        if (outputs[key] && typeof outputs[key].dispose === "function") {
          outputs[key].dispose();
        }
      }
    }
  }
}

/**
 * Post-process raw outputs for object detection.
 *
 * @param {ort.Tensor} rawTensor - Model output tensor.
 * @param {number} scoreThreshold - Threshold for confidence score.
 * @param {number} xRatio - Width scaling ratio.
 * @param {number} yRatio - Height scaling ratio.
 * @returns {Array<Object>} Array of detection results: [{bbox, classIdx, score}, ...].
 */
function postProcessDetect(rawTensor, scoreThreshold = 0.45, xRatio, yRatio) {
  const NUM_PREDICTIONS = rawTensor.dims[2];
  const NUM_BBOX_ATTRS = 4;
  const NUM_SCORES = 80;

  const predictions = rawTensor.data;

  const results = new Array(Math.min(NUM_PREDICTIONS, 100));
  let resultCount = 0;

  const bboxOffset0 = 0;
  const bboxOffset1 = NUM_PREDICTIONS;
  const bboxOffset2 = NUM_PREDICTIONS * 2;
  const bboxOffset3 = NUM_PREDICTIONS * 3;
  const scoresOffset = NUM_PREDICTIONS * NUM_BBOX_ATTRS;

  for (let i = 0; i < NUM_PREDICTIONS; i++) {
    let maxScore = 0;
    let classIdx = -1;

    const baseScoreIdx = scoresOffset + i;
    for (let c = 0; c < NUM_SCORES; c++) {
      const score = predictions[baseScoreIdx + c * NUM_PREDICTIONS];
      if (score > maxScore) {
        maxScore = score;
        classIdx = c;
      }
    }
    if (maxScore <= scoreThreshold) continue;

    const w = predictions[i + bboxOffset2] * xRatio;
    const h = predictions[i + bboxOffset3] * yRatio;
    const tlx = predictions[i + bboxOffset0] * xRatio - 0.5 * w;
    const tly = predictions[i + bboxOffset1] * yRatio - 0.5 * h;

    // if not enough space
    if (resultCount >= results.length) {
      results.length += 50;
    }

    results[resultCount++] = {
      bbox: [tlx, tly, w, h],
      classIdx,
      score: maxScore,
    };
  }

  results.length = resultCount;
  return results;
}

/**
 * Post-process for End-to-End models (like YOLO26) which output [1, 300, 6].
 * Structure: [Batch, MaxDets, [Class, Score, X1, Y1, X2, Y2]]
 *
 * @param {ort.Tensor} rawTensor - Output tensor [1, 300, 6]
 * @param {number} scoreThreshold - Confidence threshold
 * @param {number} xRatio - Width scale ratio
 * @param {number} yRatio - Height scale ratio
 * @returns {Array<Object>}
 */
function postProcessDetectEnd2End(rawTensor, scoreThreshold, xRatio, yRatio) {
  const predictions = rawTensor.data;

  // dims expected: [1, 300, 6]
  const NUM_DETECTIONS = rawTensor.dims[1]; // 300
  const NUM_ATTRIBUTES = rawTensor.dims[2]; // 6

  const results = [];

  for (let i = 0; i < NUM_DETECTIONS; i++) {
    const offset = i * NUM_ATTRIBUTES;
    const score = predictions[offset + 4];

    if (score <= scoreThreshold) break;
    // 0: x1, 1: y1, 2: x2, 3: y2
    // 4: confidence
    // 5: class_id
    const classIdx = Math.round(predictions[offset + 5]);

    const x1 = predictions[offset] * xRatio;
    const y1 = predictions[offset + 1] * yRatio;
    const x2 = predictions[offset + 2] * xRatio;
    const y2 = predictions[offset + 3] * yRatio;

    // Convert to [x, y, w, h]
    const w = x2 - x1;
    const h = y2 - y1;

    results.push({
      bbox: [x1, y1, w, h],
      classIdx: classIdx,
      score: score,
    });
  }

  return results;
}

/**
 * Post-process raw outputs for pose estimation.
 *
 * @param {ort.Tensor} rawTensor - Model output tensor.
 * @param {number} scoreThreshold - Threshold for confidence score.
 * @param {number} xRatio - Width scaling ratio.
 * @param {number} yRatio - Height scaling ratio.
 * @returns {Array<Object>} Array of pose results: [{bbox, score, keypoints}, ...].
 */
function postProcessPose(rawTensor, scoreThreshold = 0.45, xRatio, yRatio) {
  // post process
  const NUM_PREDICTIONS = rawTensor.dims[2];
  const NUM_BBOX_ATTRS = 5;
  const NUM_KEYPOINTS = 17;
  const KEYPOINT_DIMS = 3;

  const predictions = rawTensor.data;
  const bboxData = predictions.subarray(0, NUM_PREDICTIONS * NUM_BBOX_ATTRS);
  const keypointsData = predictions.subarray(NUM_PREDICTIONS * NUM_BBOX_ATTRS);
  const results = new Array();
  let resultCount = 0;

  for (let i = 0; i < NUM_PREDICTIONS; i++) {
    const score = bboxData[i + NUM_PREDICTIONS * 4];
    if (score <= scoreThreshold) continue;

    const w = bboxData[i + NUM_PREDICTIONS * 2] * xRatio;
    const h = bboxData[i + NUM_PREDICTIONS * 3] * yRatio;
    const tlx = bboxData[i] * xRatio - 0.5 * w;
    const tly = bboxData[i + NUM_PREDICTIONS] * yRatio - 0.5 * h;

    const keypoints = new Array(NUM_KEYPOINTS);
    for (let kp = 0; kp < NUM_KEYPOINTS; kp++) {
      const baseIdx = kp * KEYPOINT_DIMS * NUM_PREDICTIONS + i;
      keypoints[kp] = {
        x: keypointsData[baseIdx] * xRatio,
        y: keypointsData[baseIdx + NUM_PREDICTIONS] * yRatio,
        score: keypointsData[baseIdx + NUM_PREDICTIONS * 2],
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
 * Post-process for End-to-End Pose models.
 * Expected tensor shape: [Batch, MaxDets, 6 + Keypoints*3]
 * Structure of attributes: [x1, y1, x2, y2, score, class, kp1_x, kp1_y, kp1_conf, ...]
 *
 * @param {ort.Tensor} rawTensor - Model output tensor.
 * @param {number} scoreThreshold - Threshold for confidence score.
 * @param {number} xRatio - Width scaling ratio.
 * @param {number} yRatio - Height scaling ratio.
 * @returns {Array<Object>} Array of pose results: [{bbox, score, keypoints}, ...].
 */
function postProcessPoseEnd2End(
  rawTensor,
  scoreThreshold = 0.45,
  xRatio,
  yRatio,
) {
  // post process
  const NUM_PREDICTIONS = rawTensor.dims[1]; // 300
  const NUM_ATTRIBUTED = rawTensor.dims[2]; // 6 + 17*3
  const NUM_BBOX_ATTRS = 6; // x1, y1, x2, y2, score, classidx
  const NUM_KEYPOINTS = 17;
  const KEYPOINT_DIMS = 3; // x, y, visibility

  const predictions = rawTensor.data;
  const results = new Array();
  let resultCount = 0;

  for (let i = 0; i < NUM_PREDICTIONS; i++) {
    const offset = i * NUM_ATTRIBUTED;
    const score = predictions[offset + 4];
    if (score <= scoreThreshold) break;

    const x1 = predictions[offset] * xRatio;
    const y1 = predictions[offset + 1] * yRatio;
    const x2 = predictions[offset + 2] * xRatio;
    const y2 = predictions[offset + 3] * yRatio;

    const w = x2 - x1;
    const h = y2 - y1;

    const keypoints = new Array(NUM_KEYPOINTS);
    for (let kp = 0; kp < NUM_KEYPOINTS; kp++) {
      const baseIdx = offset + NUM_BBOX_ATTRS + kp * KEYPOINT_DIMS;
      keypoints[kp] = {
        x: predictions[baseIdx] * xRatio,
        y: predictions[baseIdx + 1] * yRatio,
        score: predictions[baseIdx + 2],
      };
    }

    results[resultCount++] = {
      bbox: [x1, y1, w, h],
      score,
      keypoints,
    };
  }
  return results;
}

/**
 * Post-process raw outputs for instance segmentation.
 *
 * @param {ort.Tensor} output0 - Detection output tensor (shape: [1, G, 4 + C + M]).
 * @param {ort.Tensor} output1 - Prototype masks output tensor (shape: [1, M, Hm, Wm]).
 * @param {number} scoreThreshold - Threshold for confidence score.
 * @param {number} xRatio - Width scaling ratio.
 * @param {number} yRatio - Height scaling ratio.
 * @returns {[Array<Object>, Object]} Tuple of [results, masksData].
 */
function postProcessSegment(
  rawTensor,
  rawMaskTensor,
  scoreThreshold,
  xRatio,
  yRatio,
) {
  const NUM_PREDICTIONS = rawTensor.dims[2];
  const NUM_BBOX_ATTRS = 4;
  const NUM_SCORES = 80;
  const NUM_MASK_WEIGHTS = 32;

  const predictions = rawTensor.data;
  const bboxData = predictions.subarray(0, NUM_PREDICTIONS * NUM_BBOX_ATTRS);
  const scoresData = predictions.subarray(
    NUM_PREDICTIONS * NUM_BBOX_ATTRS,
    NUM_PREDICTIONS * (NUM_BBOX_ATTRS + NUM_SCORES),
  );
  const maskWeightsData = predictions.subarray(
    NUM_PREDICTIONS * (NUM_BBOX_ATTRS + NUM_SCORES),
  );

  const protoMask = rawMaskTensor.data;
  const MASK_CHANNELS = rawMaskTensor.dims[1];
  const MASK_HEIGHT = rawMaskTensor.dims[2];
  const MASK_WIDTH = rawMaskTensor.dims[3];

  const results = new Array();
  let resultCount = 0;
  for (let i = 0; i < NUM_PREDICTIONS; i++) {
    let maxScore = 0;
    let classIdx = -1;

    for (let c = 0; c < NUM_SCORES; c++) {
      const score = scoresData[i + c * NUM_PREDICTIONS];
      if (score > maxScore) {
        maxScore = score;
        classIdx = c;
      }
    }
    if (maxScore <= scoreThreshold) continue;

    const w = bboxData[i + NUM_PREDICTIONS * 2] * xRatio;
    const h = bboxData[i + NUM_PREDICTIONS * 3] * yRatio;
    const tlx = bboxData[i] * xRatio - 0.5 * w;
    const tly = bboxData[i + NUM_PREDICTIONS] * yRatio - 0.5 * h;

    results[resultCount++] = {
      bbox: [tlx, tly, w, h],
      classIdx,
      score: maxScore,
      maskWeightIdx: i,
    };
  }

  const masksData = {
    protoMask,
    maskWeightsData: maskWeightsData.slice(),
    MASK_CHANNELS,
    MASK_HEIGHT,
    MASK_WIDTH,
  };

  return [results, masksData];
}

/**
 * Post-process for End-to-End Segmentation models.
 * Processes bounding boxes and extracts mask weights embedded in the main tensor.
 *
 * @param {ort.Tensor} rawTensor - Main output tensor containing boxes, scores, and mask weights.
 * @param {ort.Tensor} rawMaskTensor - Prototype masks output tensor.
 * @param {number} scoreThreshold - Threshold for confidence score.
 * @param {number} xRatio - Width scaling ratio.
 * @param {number} yRatio - Height scaling ratio.
 * @returns {[Array<Object>, Object]} Tuple of [results, masksData].
 */
function postProcessSegmentEnd2End(
  rawTensor,
  rawMaskTensor,
  scoreThreshold,
  xRatio,
  yRatio,
) {
  const NUM_PREDICTIONS = rawTensor.dims[1];
  const NUM_ATTRIBUTES = rawTensor.dims[2];
  const NUM_BBOX_ATTRS = 6; // x1, y1, x2, y2, score, classidx
  const NUM_MASK_WEIGHTS = 32;

  const predictions = rawTensor.data;

  const protoMask = rawMaskTensor.data;
  const MASK_CHANNELS = rawMaskTensor.dims[1];
  const MASK_HEIGHT = rawMaskTensor.dims[2];
  const MASK_WIDTH = rawMaskTensor.dims[3];

  const results = new Array();
  const maskWeightsData = new Float32Array(NUM_PREDICTIONS * NUM_MASK_WEIGHTS);

  let resultCount = 0;
  for (let i = 0; i < NUM_PREDICTIONS; i++) {
    const offset = i * NUM_ATTRIBUTES;
    const score = predictions[offset + 4];

    if (score <= scoreThreshold) break;

    const classIdx = Math.round(predictions[offset + 5]);
    const x1 = predictions[offset] * xRatio;
    const y1 = predictions[offset + 1] * yRatio;
    const x2 = predictions[offset + 2] * xRatio;
    const y2 = predictions[offset + 3] * yRatio;

    const w = x2 - x1;
    const h = y2 - y1;

    // copy and transpose mask weights
    for (let c = 0; c < NUM_MASK_WEIGHTS; c++) {
      const sourceIdx = offset + 6 + c;
      const destIdx = i + c * NUM_PREDICTIONS;
      maskWeightsData[destIdx] = predictions[sourceIdx];
    }

    results[resultCount++] = {
      bbox: [x1, y1, w, h],
      classIdx,
      score: score,
      maskWeightIdx: i,
    };
  }

  const masksData = {
    protoMask,
    maskWeightsData,
    MASK_CHANNELS,
    MASK_HEIGHT,
    MASK_WIDTH,
  };

  return [results, masksData];
}

/**
 * Generate mask overlay image from segmentation results.
 *
 * @param {Array<Object>} filteredResults - NMS filtered detection results.
 * @param {Object} masksData - Object containing mask prototypes and weights.
 * @param {[number, number]} overlaySize - Dimensions of the overlay [width, height].
 * @returns {ImageData|null} Resulting mask image data, or null if no results.
 */
function postProcessMask(filteredResults, masksData, overlaySize) {
  if (!filteredResults || filteredResults.length === 0) return null;
  const { protoMask, maskWeightsData, MASK_CHANNELS, MASK_HEIGHT, MASK_WIDTH } =
    masksData;

  const matsToDelete = [];

  // protoMask: [1, 32*160*160] -> cv.Mat(32, 160*160)
  const protoMaskMat = cv.matFromArray(
    MASK_CHANNELS,
    MASK_HEIGHT * MASK_WIDTH,
    cv.CV_32F,
    protoMask,
  );
  matsToDelete.push(protoMaskMat);

  try {
    // Weights x Proto_mask
    const NUM_FILTERED_RESULTS = filteredResults.length;

    const NUM_PREDICTIONS = maskWeightsData.length / MASK_CHANNELS;
    const maskWeights = new Float32Array(NUM_FILTERED_RESULTS * MASK_CHANNELS);

    for (let i = 0; i < NUM_FILTERED_RESULTS; i++) {
      const baseIdx = filteredResults[i].maskWeightIdx;
      for (let c = 0; c < MASK_CHANNELS; c++) {
        maskWeights[i * MASK_CHANNELS + c] =
          maskWeightsData[baseIdx + c * NUM_PREDICTIONS];
      }
    }

    const maskWeightsMat = cv.matFromArray(
      NUM_FILTERED_RESULTS,
      MASK_CHANNELS,
      cv.CV_32F,
      maskWeights,
    );
    matsToDelete.push(maskWeightsMat);

    const weightsMulProtoMat = new cv.Mat();
    matsToDelete.push(weightsMulProtoMat);
    const emptyMat = new cv.Mat();
    matsToDelete.push(emptyMat);
    cv.gemm(
      maskWeightsMat, // [N, 32]
      protoMaskMat, // [32, 160*160]
      1.0,
      emptyMat,
      0.0,
      weightsMulProtoMat, // [N, 160*160]
      0,
    );

    // Sigmoid
    const maskSigmoidMat = new cv.Mat();
    matsToDelete.push(maskSigmoidMat);
    const onesMat = cv.Mat.ones(
      weightsMulProtoMat.rows,
      weightsMulProtoMat.cols,
      cv.CV_32F,
    );
    matsToDelete.push(onesMat);

    const tempMat2 = new cv.Mat(
      weightsMulProtoMat.rows,
      weightsMulProtoMat.cols,
      cv.CV_32F,
      new cv.Scalar(-1),
    );
    matsToDelete.push(tempMat2);
    cv.multiply(weightsMulProtoMat, tempMat2, maskSigmoidMat);

    cv.exp(maskSigmoidMat, maskSigmoidMat);
    cv.add(maskSigmoidMat, onesMat, maskSigmoidMat);
    cv.divide(onesMat, maskSigmoidMat, maskSigmoidMat);

    // Create mask overlay
    const overlayMat = new cv.Mat(
      overlaySize[1],
      overlaySize[0],
      cv.CV_8UC4,
      new cv.Scalar(0, 0, 0, 0),
    );
    matsToDelete.push(overlayMat);

    const maskResizedMat = new cv.Mat();
    matsToDelete.push(maskResizedMat);
    const maskBinaryMat = new cv.Mat();
    matsToDelete.push(maskBinaryMat);
    const maskBinaryU8Mat = new cv.Mat();
    matsToDelete.push(maskBinaryU8Mat);

    for (let i = 0; i < NUM_FILTERED_RESULTS; i++) {
      const rowMat = maskSigmoidMat.row(i);
      matsToDelete.push(rowMat);
      const mask = rowMat.data32F;
      const maskMat = cv.matFromArray(MASK_HEIGHT, MASK_WIDTH, cv.CV_32F, mask);
      matsToDelete.push(maskMat);

      const [x, y, w, h] = filteredResults[i].bbox;

      // 1. Calculate coordinates on the 160x160 mask
      const scaleX = MASK_WIDTH / overlaySize[0];
      const scaleY = MASK_HEIGHT / overlaySize[1];

      const maskX = Math.floor(Math.max(0, x * scaleX));
      const maskY = Math.floor(Math.max(0, y * scaleY));
      const maskW = Math.ceil(Math.min(MASK_WIDTH - maskX, w * scaleX));
      const maskH = Math.ceil(Math.min(MASK_HEIGHT - maskY, h * scaleY));

      // Boundary check
      if (maskW > 0 && maskH > 0) {
        // 2. Crop the small region from 160x160 mask
        const maskRoi = maskMat.roi(new cv.Rect(maskX, maskY, maskW, maskH));
        matsToDelete.push(maskRoi);

        // 3. Resize only this small region to the target bbox size
        const targetX = Math.max(0, Math.floor(x));
        const targetY = Math.max(0, Math.floor(y));
        const targetW = Math.min(overlaySize[0] - targetX, Math.ceil(w));
        const targetH = Math.min(overlaySize[1] - targetY, Math.ceil(h));

        if (targetW > 0 && targetH > 0) {
          cv.resize(
            maskRoi,
            maskResizedMat,
            new cv.Size(targetW, targetH),
            cv.INTER_LINEAR,
          );

          // Binarize
          cv.threshold(
            maskResizedMat,
            maskBinaryMat,
            0.5,
            255,
            cv.THRESH_BINARY,
          );
          maskBinaryMat.convertTo(maskBinaryU8Mat, cv.CV_8U);

          // Colorize mask
          const color = Colors.getColor(filteredResults[i].classIdx, 0.6);
          const colorScalar = new cv.Scalar(
            color[0],
            color[1],
            color[2],
            color[3] * 255,
          );

          // Create colored mat with target size
          const maskColoredMat = new cv.Mat(
            targetH,
            targetW,
            cv.CV_8UC4,
            colorScalar,
          );
          matsToDelete.push(maskColoredMat);

          // Copy to overlay mat at the specific bbox location
          const overlayRoi = overlayMat.roi(
            new cv.Rect(targetX, targetY, targetW, targetH),
          );
          matsToDelete.push(overlayRoi);
          maskColoredMat.copyTo(overlayRoi, maskBinaryU8Mat);
        }
      }
    }

    const imgData = new ImageData(
      new Uint8ClampedArray(overlayMat.data),
      overlaySize[0],
      overlaySize[1],
    );

    return imgData;
  } catch (error) {
    console.error("Error masks:", error);
    return null;
  } finally {
    matsToDelete.forEach((mat) => {
      if (mat && !mat.isDeleted()) mat.delete();
    });
  }
}

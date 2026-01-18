import cv from "@techstark/opencv-js";
import { preProcessImage, Colors, applyNMS } from "./process-util";

/**
 * Inference pipeline for YOLO model.
 * @param {ImageData} imageData - Input image data.
 * @param {ort.InferenceSession} session - YOLO model ONNX Runtime session.
 * @param {object} modelConfig - Model configuration object.
 * @returns {Promise<object>} Inference result object containing detection results, mask image data, and inference time.
 */
export async function inferencePipeline(imageData, session, modelConfig) {
  const matsToDelete = [];
  let inputTensor = null;
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
      modelConfig.overlaySize,
      modelConfig.imgszType,
    );

    const start = performance.now();
    const outputs = await session.run({
      images: inputTensor,
    });
    output0 = outputs.output0;
    output1 = outputs.output1;
    const end = performance.now();

    // Post process
    let results, masksData, maskImgData;
    switch (modelConfig.task) {
      case "detect":
        results = postProcessDetect(
          output0,
          modelConfig.score_threshold,
          xRatio,
          yRatio,
        );
        break;
      case "pose":
        results = postProcessPose(
          output0,
          modelConfig.score_threshold,
          xRatio,
          yRatio,
        );
        break;
      case "segment":
        [results, masksData] = postProcessSegment(
          output0,
          output1,
          modelConfig.scoreThreshold,
          xRatio,
          yRatio,
        );
        break;
      default:
        console.warn(
          `Unknown task: ${modelConfig.task}, falling back to detection`,
        );
    }

    // Apply NMS
    const selectedIndices = applyNMS(
      results,
      results.map((r) => r.score),
      modelConfig.iou_threshold,
    );
    const filteredResults = selectedIndices.map((i) => results[i]);

    if (modelConfig.task === "segment") {
      maskImgData = postProcessMask(
        filteredResults,
        masksData,
        modelConfig.overlaySize,
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
    if (output0) output0.dispose();
    if (output1) output1.dispose();
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
 * Post-process raw outputs for instance segmentation.
 *
 * @param {ort.Tensor} output0 - Detection output tensor (shape: [1, G, 4 + C + M]).
 * @param {ort.Tensor} output1 - Prototype masks output tensor (shape: [1, M, Hm, Wm]).
 * @param {number} scoreThreshold - Threshold for confidence score.
 * @param {number} xRatio - Width scaling ratio.
 * @param {number} yRatio - Height scaling ratio.
 * @returns {[Array<Object>, Object]} Tuple of [results, masksData].
 */
function postProcessSegment(output0, output1, scoreThreshold, xRatio, yRatio) {
  const NUM_PREDICTIONS = output0.dims[2];
  const NUM_BBOX_ATTRS = 4;
  const NUM_SCORES = 80;
  const NUM_MASK_WEIGHTS = 32;

  const predictions = output0.data;
  const bboxData = predictions.subarray(0, NUM_PREDICTIONS * NUM_BBOX_ATTRS);
  const scoresData = predictions.subarray(
    NUM_PREDICTIONS * NUM_BBOX_ATTRS,
    NUM_PREDICTIONS * (NUM_BBOX_ATTRS + NUM_SCORES),
  );
  const maskWeightsData = predictions.subarray(
    NUM_PREDICTIONS * (NUM_BBOX_ATTRS + NUM_SCORES),
  );

  const protoMask = output1.data;
  const MASK_CHANNELS = output1.dims[1];
  const MASK_HEIGHT = output1.dims[2];
  const MASK_WIDTH = output1.dims[3];

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

  // protoMask: [1, 32*160*160] -> cv.Mat(32, 160*160)
  const protoMaskMat = cv.matFromArray(
    MASK_CHANNELS,
    MASK_HEIGHT * MASK_WIDTH,
    cv.CV_32F,
    protoMask,
  );

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

    const weightsMulProtoMat = new cv.Mat();
    cv.gemm(
      maskWeightsMat, // [N, 32]
      protoMaskMat, // [32, 160*160]
      1.0,
      new cv.Mat(),
      0.0,
      weightsMulProtoMat, // [N, 160*160]
      0,
    );

    protoMaskMat.delete();
    maskWeightsMat.delete();

    // Sigmoid
    const maskSigmoidMat = new cv.Mat();
    const onesMat = cv.Mat.ones(weightsMulProtoMat.size(), cv.CV_32F);

    const tempMat2 = new cv.Mat(
      weightsMulProtoMat.rows,
      weightsMulProtoMat.cols,
      cv.CV_32F,
      new cv.Scalar(-1),
    );
    cv.multiply(weightsMulProtoMat, tempMat2, maskSigmoidMat);
    tempMat2.delete();

    cv.exp(maskSigmoidMat, maskSigmoidMat);
    cv.add(maskSigmoidMat, onesMat, maskSigmoidMat);
    cv.divide(onesMat, maskSigmoidMat, maskSigmoidMat);

    onesMat.delete();
    weightsMulProtoMat.delete();

    // Create mask overlay
    const overlayMat = new cv.Mat(
      overlaySize[1],
      overlaySize[0],
      cv.CV_8UC4,
      new cv.Scalar(0, 0, 0, 0),
    );

    const maskResizedMat = new cv.Mat();
    const maskBinaryMat = new cv.Mat();
    const maskBinaryU8Mat = new cv.Mat();

    for (let i = 0; i < NUM_FILTERED_RESULTS; i++) {
      const mask = maskSigmoidMat.row(i).data32F;
      const maskMat = cv.matFromArray(MASK_HEIGHT, MASK_WIDTH, cv.CV_32F, mask);

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

          // Copy to overlay mat at the specific bbox location
          maskColoredMat.copyTo(
            overlayMat.roi(new cv.Rect(targetX, targetY, targetW, targetH)),
            maskBinaryU8Mat,
          );

          maskColoredMat.delete();
        }
        maskRoi.delete();
      }
      maskMat.delete();
    }
    maskResizedMat.delete();
    maskBinaryMat.delete();
    maskBinaryU8Mat.delete();
    maskSigmoidMat.delete();

    const imgData = new ImageData(
      new Uint8ClampedArray(
        overlayMat.data.buffer,
        overlayMat.data.byteOffset,
        overlayMat.data.byteLength,
      ),
      overlaySize[0],
      overlaySize[1],
    );
    overlayMat.delete();

    return imgData;
  } catch (error) {
    console.error("Error masks:", error);
    protoMaskMat.delete();
    return null;
  }
}

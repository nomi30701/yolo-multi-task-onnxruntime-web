import cv from "@techstark/opencv-js";
import { Tensor } from "onnxruntime-web/webgpu";

/**
 * Pre-process input image.
 *
 * @param {cv.Mat} srcMat - input image Mat
 * @param {[Number, Number]} size - Output size [width, height]
 * @param {String} imgszType - Processing type, "dynamic" or "zeroPad"
 * @returns {[ort.Tensor, Number, Number]} - return [inputTensor, xRatio, yRatio]
 */
const preProcessImage = (srcMat, size, imgszType) => {
  let preProcessed, xRatio, yRatio, inputTensor, divWidth, divHeight;

  if (imgszType === "dynamic") {
    [preProcessed, xRatio, yRatio, divWidth, divHeight] = imgDynamic(
      srcMat,
      size,
    );
    inputTensor = new Tensor(
      "float32",
      new Float32Array(preProcessed.data32F),
      [1, 3, divHeight, divWidth], // [batch, channel, height, width]
    );
  } else if (imgszType === "zeroPad") {
    const modelSize = [640, 640]; // yolo model default input size
    [preProcessed, xRatio, yRatio] = imgZeroPad(srcMat, modelSize, size);
    inputTensor = new Tensor(
      "float32",
      new Float32Array(preProcessed.data32F),
      [1, 3, modelSize[1], modelSize[0]], // [batch, channel, height, width]
    );
  }
  preProcessed.delete();

  return [inputTensor, xRatio, yRatio];
};

/**
 * Pre process input image.
 *
 * Zero padding to square and resize to input size.
 *
 * @param {cv.Mat} mat - Pre process yolo model input image.
 * @param {Number} modelSize - Yolo model image size input [width, height].
 * @param {Number} outputSize - Overlay image size [width, height].
 * @returns {[cv.Mat, Number, Number]} Processed input mat, xRatio, yRatio.
 */
const imgZeroPad = (mat, modelSize, outputSize) => {
  cv.cvtColor(mat, mat, cv.COLOR_RGBA2RGB);

  // Resize to dimensions divisible by 32
  const [divWidth, divHeight] = divStride(32, mat.cols, mat.rows);
  cv.resize(mat, mat, new cv.Size(divWidth, divHeight));

  // Padding to square
  const maxDim = Math.max(divWidth, divHeight);
  const rightPad = maxDim - divWidth;
  const bottomPad = maxDim - divHeight;
  cv.copyMakeBorder(
    mat,
    mat,
    0,
    bottomPad,
    0,
    rightPad,
    cv.BORDER_CONSTANT,
    new cv.Scalar(0, 0, 0),
  ); // padding to square

  // Resize to input size and normalize to [0, 1]
  const preProcessed = cv.blobFromImage(
    mat,
    1 / 255.0,
    new cv.Size(modelSize[0], modelSize[1]),
    new cv.Scalar(0, 0, 0, 0),
    false,
    false,
  );

  const xRatio = (outputSize[0] / divWidth) * (maxDim / modelSize[0]);
  const yRatio = (outputSize[1] / divHeight) * (maxDim / modelSize[1]);

  return [preProcessed, xRatio, yRatio];
};

/**
 * Pre process input image for dynamic input model.
 *
 * @param {cv.Mat} mat - Pre process yolo model input image.
 * @returns {[cv.mat, Number, Number ...]} Processed input mat, xRatio, yRatio, divWidth, divHeight.
 */
const imgDynamic = (mat, size) => {
  cv.cvtColor(mat, mat, cv.COLOR_RGBA2RGB);

  // resize image to divisible by 32
  const [divWidth, divHeight] = divStride(32, mat.cols, mat.rows);

  // resize, normalize to [0, 1]
  const preProcessed = cv.blobFromImage(
    mat,
    1 / 255.0,
    new cv.Size(divWidth, divHeight),
    new cv.Scalar(0, 0, 0, 0),
    false,
    false,
  );
  const xRatio = size[0] / divWidth; // scale factor for overlay
  const yRatio = size[1] / divHeight;
  return [preProcessed, xRatio, yRatio, divWidth, divHeight];
};

/**
 * Return height and width are divisible by stride.
 * @param {Number} stride - Stride value.
 * @param {Number} width - Image width.
 * @param {Number} height - Image height.
 * @returns {[Number]}[width, height] divisible by stride.
 **/
const divStride = (stride, width, height) => {
  width =
    width % stride >= stride / 2
      ? (Math.floor(width / stride) + 1) * stride
      : Math.floor(width / stride) * stride;

  height =
    height % stride >= stride / 2
      ? (Math.floor(height / stride) + 1) * stride
      : Math.floor(height / stride) * stride;

  return [width, height];
};

function calculateIou(box1, box2) {
  const [x1, y1, w1, h1] = box1;
  const [x2, y2, w2, h2] = box2;

  // check if boxes are valid
  if (x1 > x2 + w2 || x2 > x1 + w1 || y1 > y2 + h2 || y2 > y1 + h1) {
    return 0.0;
  }

  const box1X2 = x1 + w1;
  const box1Y2 = y1 + h1;
  const box2X2 = x2 + w2;
  const box2Y2 = y2 + h2;

  const intersectX1 = Math.max(x1, x2);
  const intersectY1 = Math.max(y1, y2);
  const intersectX2 = Math.min(box1X2, box2X2);
  const intersectY2 = Math.min(box1Y2, box2Y2);

  const intersection =
    (intersectX2 - intersectX1) * (intersectY2 - intersectY1);
  const box1Area = w1 * h1;
  const box2Area = w2 * h2;

  return intersection / (box1Area + box2Area - intersection);
}

function applyNMS(boxes, scores, iouThreshold = 0.7) {
  const n = scores.length;
  if (n === 0) return [];

  // pre calculate areas
  const areas = new Array(n);
  for (let i = 0; i < n; i++) {
    const [, , w, h] = boxes[i].bbox;
    areas[i] = w * h;
  }

  // sort indexes by scores
  const indexes = new Uint32Array(n);
  for (let i = 0; i < n; i++) indexes[i] = i;

  // sort indexes by scores in descending order
  indexes.sort((a, b) => scores[b] - scores[a]);

  // use bitmap to track suppressed boxes
  const suppress = new Uint8Array(n);
  const picked = [];

  for (let i = 0; i < n; i++) {
    const idx = indexes[i];

    if (suppress[idx]) continue;

    picked.push(idx);

    // check remaining boxes
    for (let j = i + 1; j < n; j++) {
      const otherIdx = indexes[j];

      if (suppress[otherIdx]) continue;

      const iou = calculateIou(boxes[idx].bbox, boxes[otherIdx].bbox);

      if (iou > iouThreshold) {
        suppress[otherIdx] = 1;
      }
    }
  }

  return picked;
}

/**
 * Ultralytics default color palette https://ultralytics.com/.
 *
 * This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
 * RGB values.
 */
class Colors {
  static palette = [
    "042AFF",
    "0BDBEB",
    "F3F3F3",
    "00DFB7",
    "111F68",
    "FF6FDD",
    "FF444F",
    "CCED00",
    "00F344",
    "BD00FF",
    "00B4FF",
    "DD00BA",
    "00FFFF",
    "26C000",
    "01FFB3",
    "7D24FF",
    "7B0068",
    "FF1B6C",
    "FC6D2F",
    "A2FF0B",
  ].map((c) => Colors.hex2rgba(`#${c}`));
  static n = Colors.palette.length;
  static cache = {}; // Cache for colors

  static hex2rgba(h, alpha = 1.0) {
    return [
      parseInt(h.slice(1, 3), 16),
      parseInt(h.slice(3, 5), 16),
      parseInt(h.slice(5, 7), 16),
      alpha,
    ];
  }

  static getColor(i, alpha = 1.0, bgr = false) {
    const key = `${i}-${alpha}-${bgr}`;
    if (Colors.cache[key]) {
      return Colors.cache[key];
    }
    const c = Colors.palette[i % Colors.n];
    const rgba = [...c.slice(0, 3), alpha];
    const result = bgr ? [rgba[2], rgba[1], rgba[0], rgba[3]] : rgba;
    Colors.cache[key] = result;
    return result;
  }
}

export { preProcessImage, applyNMS, Colors };

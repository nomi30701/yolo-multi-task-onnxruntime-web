import cv from "@techstark/opencv-js";
import { Tensor } from "onnxruntime-web/webgpu";

/**
 * Pre-process input image.
 *
 * @param {cv.Mat} src_mat - input image Mat
 * @param {[Number, Number]} size - Output size [width, height]
 * @param {String} imgsz_type - Processing type, "dynamic" or "zeroPad"
 * @returns {[ort.Tensor, Number, Number]} - return [input_tensor, xRatio, yRatio]
 */
const preProcess_img = (src_mat, size, imgsz_type) => {
  let preProcessed, xRatio, yRatio, input_tensor, div_width, div_height;

  if (imgsz_type === "dynamic") {
    [preProcessed, xRatio, yRatio, div_width, div_height] = img_dynamic(
      src_mat,
      size
    );
    input_tensor = new Tensor(
      "float32",
      preProcessed.data32F,
      [1, 3, div_height, div_width] // [batch, channel, height, width]
    );
  } else if (imgsz_type === "zeroPad") {
    const model_size = [640, 640]; // yolo model default input size
    [preProcessed, xRatio, yRatio] = img_zeroPad(src_mat, model_size, size);
    input_tensor = new Tensor(
      "float32",
      preProcessed.data32F,
      [1, 3, model_size[1], model_size[0]] // [batch, channel, height, width]
    );
  }
  preProcessed.delete();

  return [input_tensor, xRatio, yRatio];
};

/**
 * Pre process input image.
 *
 * Zero padding to square and resize to input size.
 *
 * @param {cv.Mat} mat - Pre process yolo model input image.
 * @param {Number} model_size - Yolo model image size input [width, height].
 * @param {Number} output_size - Overlay image size [width, height].
 * @returns {[cv.Mat, Number, Number]} Processed input mat, xRatio, yRatio.
 */
const img_zeroPad = (mat, model_size, output_size) => {
  cv.cvtColor(mat, mat, cv.COLOR_RGBA2RGB);

  // Resize to dimensions divisible by 32
  const [div_width, div_height] = divStride(32, mat.cols, mat.rows);
  cv.resize(mat, mat, new cv.Size(div_width, div_height));

  // Padding to square
  const max_dim = Math.max(div_width, div_height);
  const right_pad = max_dim - div_width;
  const bottom_pad = max_dim - div_height;
  cv.copyMakeBorder(
    mat,
    mat,
    0,
    bottom_pad,
    0,
    right_pad,
    cv.BORDER_CONSTANT,
    new cv.Scalar(0, 0, 0)
  ); // padding to square

  // Resize to input size and normalize to [0, 1]
  const preProcessed = cv.blobFromImage(
    mat,
    1 / 255.0,
    { width: model_size[0], height: model_size[1] },
    [0, 0, 0, 0],
    false,
    false
  );

  const xRatio = (output_size[0] / div_width) * (max_dim / model_size[0]);
  const yRatio = (output_size[1] / div_height) * (max_dim / model_size[1]);

  return [preProcessed, xRatio, yRatio];
};

/**
 * Pre process input image for dynamic input model.
 *
 * @param {cv.Mat} mat - Pre process yolo model input image.
 * @returns {[cv.mat, Number, Number ...]} Processed input mat, xRatio, yRatio, div_width, div_height.
 */
const img_dynamic = (mat, size) => {
  cv.cvtColor(mat, mat, cv.COLOR_RGBA2RGB);

  // resize image to divisible by 32
  const [div_width, div_height] = divStride(32, mat.cols, mat.rows);

  // resize, normalize to [0, 1]
  const preProcessed = cv.blobFromImage(
    mat,
    1 / 255.0,
    { width: div_width, height: div_height },
    [0, 0, 0, 0],
    false,
    false
  );
  const xRatio = size[0] / div_width; // scale factor for overlay
  const yRatio = size[1] / div_height;
  return [preProcessed, xRatio, yRatio, div_width, div_height];
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

function calculateIOU(box1, box2) {
  const [x1, y1, w1, h1] = box1;
  const [x2, y2, w2, h2] = box2;

  // check if boxes are valid
  if (x1 > x2 + w2 || x2 > x1 + w1 || y1 > y2 + h2 || y2 > y1 + h1) {
    return 0.0;
  }

  const box1_x2 = x1 + w1;
  const box1_y2 = y1 + h1;
  const box2_x2 = x2 + w2;
  const box2_y2 = y2 + h2;

  const intersect_x1 = Math.max(x1, x2);
  const intersect_y1 = Math.max(y1, y2);
  const intersect_x2 = Math.min(box1_x2, box2_x2);
  const intersect_y2 = Math.min(box1_y2, box2_y2);

  const intersection =
    (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1);
  const box1_area = w1 * h1;
  const box2_area = w2 * h2;

  return intersection / (box1_area + box2_area - intersection);
}

function applyNMS(boxes, scores, iou_threshold = 0.7) {
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

      const iou = calculateIOU(boxes[idx].bbox, boxes[otherIdx].bbox);

      if (iou > iou_threshold) {
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

export { preProcess_img, applyNMS, Colors };

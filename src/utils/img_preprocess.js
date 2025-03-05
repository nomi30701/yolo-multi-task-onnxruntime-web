/**
 * Pre process input image.
 *
 * Resize and normalize image.
 *
 *
 * @param {cv.Mat} mat - Pre process yolo model input image.
 * @param {Number} input_width - Yolo model input width.
 * @param {Number} input_height - Yolo model input height.
 * @returns {cv.Mat} Processed input mat.
 */
const preProcess = (mat, input_width, input_height) => {
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
  );

  // Calculate ratios
  const xRatio = mat.cols / input_width;
  const yRatio = mat.rows / input_height;

  // Resize to input dimensions and normalize to [0, 1]
  const preProcessed = cv.blobFromImage(
    mat,
    1 / 255.0,
    new cv.Size(input_width, input_height),
    new cv.Scalar(0, 0, 0),
    false,
    false
  );

  return [preProcessed, xRatio, yRatio];
};

/**
 * Pre process input image.
 *
 * Normalize image.
 *
 * @param {cv.Mat} mat - Pre process yolo model input image.
 * @param {Number} input_width - Yolo model input width.
 * @param {Number} input_height - Yolo model input height.
 * @returns {cv.Mat} Processed input mat.
 */
const preProcess_dynamic = (mat) => {
  cv.cvtColor(mat, mat, cv.COLOR_RGBA2RGB);

  // resize image to divisible by 32
  const [div_width, div_height] = divStride(32, mat.cols, mat.rows);
  // resize, normalize to [0, 1]
  const preProcessed = cv.blobFromImage(
    mat,
    1 / 255.0,
    new cv.Size(div_width, div_height),
    new cv.Scalar(0, 0, 0),
    false,
    false
  );
  return [preProcessed, div_width, div_height];
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

  const box1_x2 = x1 + w1;
  const box1_y2 = y1 + h1;
  const box2_x2 = x2 + w2;
  const box2_y2 = y2 + h2;

  const intersect_x1 = Math.max(x1, x2);
  const intersect_y1 = Math.max(y1, y2);
  const intersect_x2 = Math.min(box1_x2, box2_x2);
  const intersect_y2 = Math.min(box1_y2, box2_y2);

  if (intersect_x2 <= intersect_x1 || intersect_y2 <= intersect_y1) {
    return 0.0;
  }

  const intersection =
    (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1);
  const box1_area = w1 * h1;
  const box2_area = w2 * h2;

  return intersection / (box1_area + box2_area - intersection);
}

function applyNMS(boxes, scores, iou_threshold = 0.35) {
  const picked = [];
  const indexes = Array.from(Array(scores.length).keys());

  indexes.sort((a, b) => scores[b] - scores[a]);

  while (indexes.length > 0) {
    const current = indexes[0];
    picked.push(current);

    const rest = indexes.slice(1);
    indexes.length = 0;

    for (const idx of rest) {
      const iou = calculateIOU(boxes[current].bbox, boxes[idx].bbox);
      if (iou <= iou_threshold) {
        indexes.push(idx);
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

export { preProcess, preProcess_dynamic, applyNMS, Colors };

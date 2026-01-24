import { Colors } from "./process-util";

const SKELETON = [
  [15, 13],
  [13, 11],
  [16, 14],
  [14, 12],
  [11, 12],
  [5, 11],
  [6, 12],
  [5, 6],
  [5, 7],
  [6, 8],
  [7, 9],
  [8, 10],
  [1, 2],
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
  [3, 5],
  [4, 6],
];

/**
 * Main entry point to render detection, segmentation, or pose results onto the canvas.
 * Dispatches drawing tasks based on the specific model task type.
 *
 * @param {Array<Object>} predictions - Array of prediction results (bboxes, scores, keypoints, etc.).
 * @param {ImageData|null} maskImageData - Processed segmentation mask image data (only for 'segment' task).
 * @param {CanvasRenderingContext2D} overlayCtx - The 2D context of the overlay canvas.
 * @param {string} task - The YOLO task type: "detect", "pose", or "segment".
 * @param {Object} classes - Object containing class names mapping (e.g., classes[0] = 'person').
 */
export async function renderOverlay(
  predictions,
  maskImageData,
  overlayCtx,
  task,
  classes,
) {
  if (!predictions || predictions.length === 0) return;

  // Calculate diagonal length of the canvas
  const diagonalLength = Math.hypot(
    overlayCtx.canvas.width,
    overlayCtx.canvas.height,
  );
  const lineWidth = diagonalLength / 250;

  // Draw based on task type
  switch (task) {
    case "pose":
      drawPoseEstimation(predictions, overlayCtx, lineWidth);
      break;
    case "seg":
      if (maskImageData) {
        overlayCtx.putImageData(maskImageData, 0, 0);
      }
      drawBoundingBoxes(predictions, overlayCtx, lineWidth, classes);
      break;
    case "detect":
      drawBoundingBoxes(predictions, overlayCtx, lineWidth, classes);
      break;
  }
}

/**
 * Draws bounding boxes and labels for object detection.
 * Optimizes performance by grouping predictions by class to minimize canvas state changes (fillStyle/strokeStyle).
 *
 * @param {Array<Object>} predictions - Array of detection objects {bbox: [x, y, w, h], classIdx, score}.
 * @param {CanvasRenderingContext2D} ctx - The canvas 2D context.
 * @param {number} lineWidth - Thickness of the bounding box lines.
 * @param {Object} classes - Object containing class names mapping.
 */
function drawBoundingBoxes(predictions, ctx, lineWidth, classes) {
  if (!predictions || predictions.length === 0) return;

  const predictionsByClass = {};

  predictions.forEach((predict) => {
    const classId = predict.classIdx;
    if (!predictionsByClass[classId]) predictionsByClass[classId] = [];
    predictionsByClass[classId].push(predict);
  });

  Object.entries(predictionsByClass).forEach(([classId, items]) => {
    const color = Colors.getColor(Number(classId), 0.2);
    const borderColor = Colors.getColor(Number(classId), 0.8);
    const rgbaFillColor = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${color[3]})`;
    const rgbaBorderColor = `rgba(${borderColor[0]}, ${borderColor[1]}, ${borderColor[2]}, ${borderColor[3]})`;
    const labelColor = `rgba(${borderColor[0]}, ${borderColor[1]}, ${borderColor[2]}, 1.0)`;

    // 1. filled boxes (Fill)
    ctx.fillStyle = rgbaFillColor;
    ctx.beginPath();
    items.forEach((predict) => {
      const [x1, y1, width, height] = predict.bbox;
      ctx.rect(x1, y1, width, height);
    });
    ctx.fill();

    // 2. draw edge line (Stroke)
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = rgbaBorderColor;
    ctx.beginPath();
    items.forEach((predict) => {
      const [x1, y1, width, height] = predict.bbox;
      ctx.rect(x1, y1, width, height);
    });
    ctx.stroke();

    // 3. draw text
    ctx.font = "16px Arial";

    items.forEach((predict) => {
      const [x1, y1] = predict.bbox;
      const text = `${classes.classes[predict.classIdx]} ${predict.score.toFixed(2)}`;

      drawTextWithBackground(ctx, text, x1, y1, labelColor);
    });
  });
}

/**
 * Draws pose estimation results including bounding boxes, skeleton lines, and keypoints.
 * Filtering is applied to ensure only keypoints with high confidence (> 0.5) are drawn.
 *
 * @param {Array<Object>} predictions - Array of pose objects {bbox, score, keypoints: [{x, y, score}, ...]}.
 * @param {CanvasRenderingContext2D} ctx - The canvas 2D context.
 * @param {number} lineWidth - Thickness of lines.
 */
function drawPoseEstimation(predictions, ctx, lineWidth) {
  if (!predictions || predictions.length === 0) return;

  // 1. draw bounding boxes
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = "green";
  ctx.beginPath();
  predictions.forEach((predict) => {
    const [x1, y1, width, height] = predict.bbox;
    ctx.rect(x1, y1, width, height);
  });
  ctx.stroke();

  // 2. draw score text
  ctx.font = "16px Arial";
  predictions.forEach((predict) => {
    const [x1, y1] = predict.bbox;
    const text = `score ${predict.score.toFixed(2)}`;

    drawTextWithBackground(ctx, text, x1, y1, "rgba(0, 128, 0, 0.8)");
  });

  // 3. draw skeleton
  ctx.strokeStyle = "rgb(255, 165, 0)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  predictions.forEach((predict) => {
    if (!predict.keypoints) return;
    SKELETON.forEach(([i, j]) => {
      const kp1 = predict.keypoints[i];
      const kp2 = predict.keypoints[j];
      if (kp1 && kp2 && kp1.score > 0.5 && kp2.score > 0.5) {
        ctx.moveTo(kp1.x, kp1.y);
        ctx.lineTo(kp2.x, kp2.y);
      }
    });
  });
  ctx.stroke();

  // 4. draw keypoints
  ctx.fillStyle = "red";
  ctx.beginPath();
  predictions.forEach((predict) => {
    if (!predict.keypoints) return;
    predict.keypoints.forEach((keypoint) => {
      const { x, y, score } = keypoint;
      if (score < 0.5) return;
      ctx.moveTo(x + 3, y);
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
    });
  });
  ctx.fill();
}

/**
 * Utility function to draw text with a solid background rectangle.
 * Ensures text is readable against any image background.
 * Automatically flips the label position if it's too close to the top edge.
 *
 * @param {CanvasRenderingContext2D} ctx - The canvas 2D context.
 * @param {string} text - The string to display.
 * @param {number} x - The X coordinate (usually left side of bbox).
 * @param {number} y - The Y coordinate (usually top side of bbox).
 * @param {string} [backgroundColor="black"] - The background color string (default: black).
 */
function drawTextWithBackground(ctx, text, x, y, backgroundColor = "black") {
  const textHeight = 16;
  const padding = 4;

  // 直接測量，現代瀏覽器優化得很好，Cache 這裡帶來的風險大於收益
  const textWidth = ctx.measureText(text).width;

  let textY = y - 5;
  let rectY = y - textHeight - 4;

  if (rectY < 0) {
    textY = y + textHeight + 5;
    rectY = y + 1;
  }

  const prevFillStyle = ctx.fillStyle;

  // 畫背景
  ctx.fillStyle = backgroundColor;
  ctx.fillRect(x - 1, rectY, textWidth + padding, textHeight + padding);

  // 畫文字
  ctx.fillStyle = "white";
  ctx.fillText(text, x, textY);

  // 恢復樣式
  ctx.fillStyle = prevFillStyle;
}

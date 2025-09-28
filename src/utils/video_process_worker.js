import cv from "@techstark/opencv-js";
import { InferenceSession } from "onnxruntime-web/webgpu";
import { MP4Demuxer } from "./demuxer";
import { Muxer, ArrayBufferTarget } from "mp4-muxer";
import { inference_pipeline } from "./inference_pipeline";
import { render_overlay } from "./render_overlay";

self.onmessage = async function (e) {
  const { file, modelConfig } = e.data;

  // Model
  const yolo_model = await InferenceSession.create(modelConfig.model_path, {
    executionProviders: [modelConfig.backend],
  });

  // State variables
  let inputCanvas, inputCtx, resultCanvas, resultCtx;
  let decoder = null;
  let encoder = null;
  let muxer = null;
  let frameCount = 0;
  let totalFrames = 0;

  // Frame queue for processing
  let frameQueue = [];
  let isProcessing = true;

  const onConfig = (config) => {
    totalFrames = config.nb_frames;

    inputCanvas = new OffscreenCanvas(config.codedWidth, config.codedHeight);
    resultCanvas = new OffscreenCanvas(config.codedWidth, config.codedHeight);
    inputCtx = inputCanvas.getContext("2d", {
      willReadFrequently: true,
    });
    resultCtx = resultCanvas.getContext("2d", {
      willReadFrequently: true,
    });

    // Initialize Muxer
    muxer = new Muxer({
      target: new ArrayBufferTarget(),
      video: {
        codec: "avc", // H.264
        width: config.codedWidth,
        height: config.codedHeight,
      },
      fastStart: "in-memory",
      firstTimestampBehavior: "offset",
    });

    // Initialize Encoder
    encoder = new VideoEncoder({
      output: (chunk, meta) => {
        muxer.addVideoChunk(chunk, meta);
      },
      error: (e) => {
        console.error("Encoder Error: ", e);
        self.postMessage({ statusMsg: `Encoder Error: ${e.message}` });
      },
    });

    encoder.configure({
      codec: "avc1.640028", // H.264
      width: config.codedWidth,
      height: config.codedHeight,
      bitrate: config.bitrate || 2_000_000, // 2Mbps
    });

    // Initialize Decoder
    decoder = new VideoDecoder({
      output: (frame) => {
        frameQueue.push(frame);
        if (isProcessing) {
          processNextFrame();
        }
      },
      error: (e) => {
        console.error("Decoder Error:", e);
        self.postMessage({ statusMsg: `Decoder Error: ${e.message}` });
      },
    });

    decoder.configure(config);

    self.postMessage({
      statusMsg: "✅ Initialize End, Start process...",
    });
  };

  // Process video chunks
  const onChunk = (chunk) => {
    if (decoder && decoder.state === "configured") {
      decoder.decode(chunk);
    } else {
      console.error("Encoder not ready");
      self.postMessage({ statusMsg: "Encoder not ready" });
    }
  };

  // Frame process function
  async function processNextFrame() {
    isProcessing = frameQueue.length === 0;
    if (isProcessing) {
      if (decoder.decodeQueueSize === 0) {
        finalizeVideo();
      }
      return;
    }

    const frame = frameQueue.shift();
    try {
      inputCtx.drawImage(frame, 0, 0);
      resultCtx.drawImage(frame, 0, 0);

      // Process input frame
      // const imgData = inputCtx.getImageData(
      //   0,
      //   0,
      //   inputCanvas.width,
      //   inputCanvas.height
      // );
      // const src_mat = cv.matFromImageData(imgData);

      // Inference, Draw
      const [results, inferenceTime] = await inference_pipeline(
        inputCanvas,
        yolo_model,
        [inputCanvas.width, inputCanvas.height],
        modelConfig
      );
      await render_overlay(results, modelConfig.task, resultCtx);

      // Create frame from result canvas
      const outputFrame = new VideoFrame(resultCanvas, {
        timestamp: frame.timestamp,
        duration: frame.duration,
      });

      // Encode output frame
      encoder.encode(outputFrame);
      outputFrame.close();
      frameCount++;

      self.postMessage({
        statusMsg: `${
          Math.floor(Date.now() / 1000) % 2 === 0 ? "⚫" : "🔴"
        } Processing - ${frameCount}/${
          totalFrames || "Unknow"
        } (${inferenceTime}ms)`,
        progress: totalFrames > 0 ? frameCount / totalFrames : 0,
      });
    } catch (e) {
      console.error("Frame process error:", e);
      self.postMessage({ statusMsg: `Frame process error: ${e.message}` });
    } finally {
      frame.close();
      processNextFrame();
    }
  }

  // Finalize video processing
  async function finalizeVideo() {
    try {
      self.postMessage({ statusMsg: "🔄 Finalize Video Encoding..." });

      if (decoder && decoder.state === "configured") {
        await decoder.flush();
        await decoder.close();
        return;
      }
      if (encoder && encoder.state === "configured") {
        await encoder.flush();
        await encoder.close();
      }

      // Video muxer finalize
      if (muxer) {
        muxer.finalize();
        const buffer = muxer.target.buffer;
        const blob = new Blob([buffer], { type: "video/mp4" });
        inputCanvas = null;

        self.postMessage({
          statusMsg: "✅ Video Processing Complete!",
          processedVideo: blob,
        });
      }
    } catch (e) {
      console.error("Video Processing Error:", e);
      self.postMessage({ statusMsg: `Video Processing Error: ${e.message}` });
    }
  }

  // Start demuxer
  try {
    new MP4Demuxer(file, onConfig, onChunk);
    self.postMessage({ statusMsg: "🔄 Start demuxer..." });
  } catch (e) {
    console.error("Demuxer Initialize Error:", e);
    self.postMessage({ statusMsg: `Demuxer Initialize Error: ${e.message}` });
  }
};

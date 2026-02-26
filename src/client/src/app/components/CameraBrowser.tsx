"use client";

import React, {
  useRef,
  useEffect,
  useCallback,
  useState,
  forwardRef,
  useImperativeHandle,
} from "react";
import { Hands, HAND_CONNECTIONS } from "@mediapipe/hands";
import * as cam from "@mediapipe/camera_utils";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import socket from "../socket";

const EMIT_INTERVAL_MS = 60; // ~16 fps max to server

/** Exposed handle so parent can pause / resume detection */
export interface CameraBrowserHandle {
  pause: () => void;
  resume: () => void;
  isPaused: () => boolean;
}

const CameraBrowser = forwardRef<CameraBrowserHandle>((_, ref) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const lastEmitRef = useRef<number>(0);
  const pausedRef = useRef<boolean>(false);
  const cameraRef = useRef<cam.Camera | null>(null);
  const [paused, setPaused] = useState(false);

  // Live overlay state — updated via server response
  const overlayRef = useRef<{ letter: string; confidence: number } | null>(null);

  // Listen for live classification results from server
  useEffect(() => {
    const handler = (data: { letter: string; confidence: number }) => {
      overlayRef.current = data;
    };
    socket.on("R-LIVE-LETTER", handler);
    return () => {
      socket.off("R-LIVE-LETTER", handler);
    };
  }, []);

  // Expose pause/resume to parent
  useImperativeHandle(ref, () => ({
    pause() {
      pausedRef.current = true;
      setPaused(true);
    },
    resume() {
      pausedRef.current = false;
      setPaused(false);
    },
    isPaused() {
      return pausedRef.current;
    },
  }));

  const onResults = useCallback((results: any) => {
    const canvas = canvasRef.current;
    const ctx = ctxRef.current;
    if (!canvas || !ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

    // If paused, just draw the camera feed — don't process or emit
    if (pausedRef.current) {
      // Draw paused indicator
      ctx.fillStyle = "rgba(0,0,0,0.45)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.font = "bold 48px Arial";
      ctx.fillStyle = "#FFFFFF";
      ctx.textAlign = "center";
      ctx.fillText("PAUSED", canvas.width / 2, canvas.height / 2);
      ctx.textAlign = "start";
      return;
    }

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      for (const landmarks of results.multiHandLandmarks) {
        drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
          color: "#00FF00",
          lineWidth: 3,
        });
        drawLandmarks(ctx, landmarks, {
          color: "#FF0000",
          lineWidth: 1,
          radius: 4,
        });
      }

      // ── Live letter + confidence overlay (#4) ──
      const overlay = overlayRef.current;
      if (overlay) {
        const pct = Math.round(overlay.confidence * 100 * 100) / 100;
        const text = `${overlay.letter}  ${pct}%`;

        // Position near the wrist of the first hand
        const wrist = results.multiHandLandmarks[0][0];
        const ox = Math.max(10, wrist.x * canvas.width - 60);
        const oy = Math.max(50, wrist.y * canvas.height + 60);

        // Background pill
        ctx.font = "bold 36px Arial";
        const metrics = ctx.measureText(text);
        ctx.fillStyle = "rgba(0,0,0,0.65)";
        ctx.roundRect(ox - 8, oy - 34, metrics.width + 16, 44, 8);
        ctx.fill();

        // Text
        ctx.fillStyle = pct >= 80 ? "#00FF88" : "#FFAA00";
        ctx.fillText(text, ox, oy);
      }

      // ── Throttle emissions to server ──
      const now = Date.now();
      if (now - lastEmitRef.current >= EMIT_INTERVAL_MS) {
        lastEmitRef.current = now;
        const handLandmarks = results.multiHandLandmarks[0];
        const landmarkArray = handLandmarks.map((lm: any) => [
          lm.x,
          lm.y,
          lm.z,
        ]);

        // Detect handedness (#8) — MediaPipe returns it in multiHandedness
        let handedness = "right";
        if (
          results.multiHandedness &&
          results.multiHandedness.length > 0 &&
          results.multiHandedness[0]?.label
        ) {
          // MediaPipe mirrors the label for webcam — "Left" in results means user's right hand
          handedness = results.multiHandedness[0].label === "Left"
            ? "right"
            : "left";
        }

        socket.emit("hand-landmarks", {
          landmarks: landmarkArray,
          handedness,
        });
      }
    } else {
      overlayRef.current = null; // clear overlay when no hand
      socket.emit("no-hand-detected");
    }
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    canvas.style.width = "100%";
    ctxRef.current = canvas.getContext("2d");

    const hands = new Hands({
      locateFile: (file) => `/landmarker/${file}`,
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 0,
      minDetectionConfidence: 0.75,
      minTrackingConfidence: 0.75,
    });

    hands.onResults(onResults);

    const camera = new cam.Camera(video, {
      onFrame: async () => {
        await hands.send({ image: video });
      },
      width: 960,
      height: 540,
    });
    camera.start();
    cameraRef.current = camera;

    return () => {
      camera.stop();
    };
  }, [onResults]);

  return (
    <div className="w-full border-b border-white border-opacity-20 overflow-hidden relative">
      <video ref={videoRef} style={{ display: "none" }} />
      <canvas ref={canvasRef} width="960" height="540" className="w-full" />
    </div>
  );
});

CameraBrowser.displayName = "CameraBrowser";
export default CameraBrowser;

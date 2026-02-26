"use client";

import React from "react";
import "regenerator-runtime/runtime";
import { useEffect, useRef, useState } from "react";
import SpeechRecognition, {
  useSpeechRecognition,
} from "react-speech-recognition";

import Camera, { CameraBrowserHandle } from "./components/CameraBrowser";
import { Slider } from "@/ui/components/Slider";
import Checkbox from "@/ui/components/Checkbox";
import Transcription from "./components/Transcription";
import Visualization from "./components/Visualization";
import socket from "./socket";

export default function Home() {
  const wordAnimationsToPlay = useRef<any>([]);
  const [currentWord, setCurrentWord] = useState<string>("");
  const { transcript, resetTranscript, listening } = useSpeechRecognition();
  const [signingSpeed, setSigningSpeed] = useState<number>(50);
  const [ASLTranscription, setASLTranscription] = useState("");

  // Feature state
  const cameraRef = useRef<CameraBrowserHandle>(null);
  const [cameraPaused, setCameraPaused] = useState(false);
  const [autocorrect, setAutocorrect] = useState(true);

  // ── Socket listeners ──
  useEffect(() => {
    socket.on("connect", () => {
      console.log("Connected to server");
    });

    socket.on("R-TRANSCRIPTION", (data) => {
      setASLTranscription(data);
    });

    socket.on("E-ANIMATION", (animations) => {
      wordAnimationsToPlay.current = [
        ...wordAnimationsToPlay.current,
        ...animations,
      ];
    });
  }, []);

  // ── Send autocorrect preference to server whenever it changes ──
  useEffect(() => {
    socket.emit("set-autocorrect", { enabled: autocorrect });
  }, [autocorrect]);

  // ── Speech → ASL animation ──
  useEffect(() => {
    const timeout = setTimeout(() => {
      socket.emit("E-REQUEST-ANIMATION", transcript.toLowerCase());
      resetTranscript();
    }, 2000);

    return () => {
      clearTimeout(timeout);
    };
  }, [transcript]);

  function getNextWord(): string | null {
    if (!wordAnimationsToPlay.current.length) {
      return null;
    }
    let animation = wordAnimationsToPlay.current.shift();
    setCurrentWord(animation[0]);
    return animation[1];
  }

  function clear() {
    socket.emit("R-CLEAR-TRANSCRIPTION");
    setASLTranscription("");
  }

  // ── Pause / Resume webcam (#1) ──
  function toggleCamera() {
    if (cameraRef.current?.isPaused()) {
      cameraRef.current.resume();
      setCameraPaused(false);
    } else {
      cameraRef.current?.pause();
      setCameraPaused(true);
    }
  }

  // ── Stop / Start listening (#2) ──
  function toggleListening() {
    if (listening) {
      SpeechRecognition.stopListening();
    } else {
      SpeechRecognition.startListening({ continuous: true });
    }
  }

  return (
    <div className="w-screen h-screen flex flex-row gap-4 p-4">
      {/* ════ LEFT PANEL: ASL Fingerspell → English ════ */}
      <div className="flex flex-col gap-4 items-center grow">
        <h1 className="text-2xl text-white">Fingerspell → English</h1>
        <div className="border w-full h-full flex-col flex rounded">
          <Camera ref={cameraRef} />
          <Transcription content={ASLTranscription} />
          <div className="py-4 px-4 flex items-center justify-end gap-4 bg-white bg-opacity-10">
            <Checkbox
              label="Autocorrect"
              checked={autocorrect}
              onChange={setAutocorrect}
            />
            <div
              onClick={toggleCamera}
              className={`px-4 py-1 border-opacity-20 border rounded transition duration-300 cursor-pointer ${
                cameraPaused
                  ? "bg-green-600 bg-opacity-80 border-green-400 hover:bg-green-500"
                  : "bg-red-600 bg-opacity-80 border-red-400 hover:bg-red-500"
              }`}
            >
              <p className="text-white text-lg select-none">
                {cameraPaused ? "Resume" : "Pause"}
              </p>
            </div>
            <div
              onClick={clear}
              className="px-4 py-1 border-white border-opacity-20 border rounded hover:bg-white hover:bg-opacity-10 transition duration-300 cursor-pointer"
            >
              <p className="text-white text-lg select-none">Clear</p>
            </div>
          </div>
        </div>
      </div>

      {/* ════ RIGHT PANEL: English → ASL ════ */}
      <div className="flex flex-col gap-4 items-center grow">
        <h1 className="text-2xl text-white">English → Fingerspell</h1>
        <div className="border w-full h-full flex-col flex rounded">
          <Visualization
            signingSpeed={signingSpeed}
            getNextWord={getNextWord}
            currentWord={currentWord}
          />
          <Transcription content={transcript} />
          <div className="py-4 px-4 flex flex-col items-start gap-2 bg-white bg-opacity-10">
            <div className="flex items-center justify-between w-full">
              <p className="text-lg text-white">Signing Speed</p>
              <div className="flex items-center gap-4">
                <div
                  onClick={toggleListening}
                  className={`px-4 py-1 border-opacity-20 border rounded transition duration-300 cursor-pointer ${
                    listening
                      ? "bg-red-600 bg-opacity-80 border-red-400 hover:bg-red-500"
                      : "bg-green-600 bg-opacity-80 border-green-400 hover:bg-green-500"
                  }`}
                >
                  <p className="text-white select-none">
                    {listening ? "Stop" : "Start"}
                  </p>
                </div>
                <Checkbox label="ASL Gloss" />
              </div>
            </div>

            <Slider
              defaultValue={[signingSpeed]}
              value={[signingSpeed]}
              onValueChange={(value) => setSigningSpeed(value[0])}
              min={20}
              max={100}
              step={1}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

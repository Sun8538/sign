import { useEffect, useRef } from "react";
import { useThree, useFrame } from "@react-three/fiber";

import { drawPoint, connectHands, connectPose } from "./lib";

export default function Avatar({
  signingSpeed,
  getNextWord,
}: {
  signingSpeed: number;
  getNextWord: () => any;
}) {
  const { camera } = useThree();

  const start_time = useRef(0);
  const word = useRef<any>(null);
  const previous_frame = useRef(0);

  useFrame(({ clock, scene }) => {
    const elapsed = clock.getElapsedTime() - start_time.current;
    const frame_index = Math.floor(elapsed * signingSpeed);

    if (frame_index !== previous_frame.current) {
      previous_frame.current = frame_index;

      // Get the next word
      if (!word.current) {
        word.current = getNextWord();
        start_time.current = clock.getElapsedTime();
        previous_frame.current = 0;
        return;
      }

      // Reset the word
      if (frame_index >= word.current.length) {
        word.current = null;
        return;
      }

      //   Clear previous frame
      scene.remove(...scene.children);

      const left: number[][] = word.current[0][2][0];
      const right: number[][] = word.current[0][2][1];
      const pose: number[][] = word.current[0][1];

      pose.map((point) => drawPoint(point[1], point[2], point[3]));

      left.map((point) => drawPoint(point[1], point[2], point[3]));

      right.map((point) => drawPoint(point[1], point[2], point[3]));

      connectHands(frame_index, word.current, scene);
      connectPose(frame_index, word.current, scene);
    }
  });

  useEffect(() => {
    // Position camera to view the avatar better
    // Data coordinates are roughly x: 100-600, y: 180-1400 (scaled by 0.015)
    // So avatar is at approximately x: 1.5-9, y: -21 to -2.7
    camera.position.set(7, -8, 15);
    camera.lookAt(7, -10, 0);
  }, [camera]);

  return null;
}

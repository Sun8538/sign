import os
import cv2
import numpy as np
import mediapipe as mp
from cv2.typing import MatLike

LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"

# Use the new mediapipe tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
RunningMode = mp.tasks.vision.RunningMode
drawing_utils = mp.tasks.vision.drawing_utils
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections


class Landmarker:

    def __init__(
        self,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.75,
        min_tracking_confidence: float = 0.75,
        max_num_hands: int = 1,
    ):
        # Find the hand_landmarker.task model file
        _utils_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(_utils_dir, "hand_landmarker.task")

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.model = HandLandmarker.create_from_options(options)

    def draw_landmarks(self, image: MatLike):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        results = self.model.detect(mp_image)

        if not results.hand_landmarks:
            return False, image, None, None, None

        # Draw landmarks on the image
        for hand_landmarks in results.hand_landmarks:
            # Convert NormalizedLandmark list to proto-like format for drawing
            from mediapipe.framework.formats import landmark_pb2
            landmark_proto = landmark_pb2.NormalizedLandmarkList()
            for lm in hand_landmarks:
                landmark_proto.landmark.add(x=lm.x, y=lm.y, z=lm.z)

            drawing_utils.draw_landmarks(
                image,
                landmark_proto,
                HandLandmarksConnections.HAND_CONNECTIONS,
                drawing_utils.DrawingSpec(
                    color=(0, 0, 255), thickness=8, circle_radius=8
                ),
                drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=6, circle_radius=2
                ),
            )

        hand = results.hand_landmarks[0]
        points = self.normalize_points(
            np.array(
                [(lm.x, lm.y, lm.z) for lm in hand]
            )
        )
        handedness = results.handedness[0][0].category_name.lower()

        return (
            True,
            image,
            points,
            (hand[0].x, hand[0].y),
            handedness,
        )

    def normalize_points(self, points):
        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])
        for i in range(len(points)):
            points[i][0] = (points[i][0] - min_x) / (max_x - min_x)
            points[i][1] = (points[i][1] - min_y) / (max_y - min_y)

        points = np.expand_dims(points, axis=0)

        return points

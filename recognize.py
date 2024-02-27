import threading
import time

import cv2
import numpy as np
import torch
import yaml
from torchvision import transforms

from face_alignment.alignment import norm_crop
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
from face_tracking.tracker.byte_tracker import BYTETracker
from face_tracking.tracker.visualize import plot_tracking
from face_detector.scrfd.detector import SCRFD

class FaceControl:
    def __init__(self, detector_path, tracking_path, recognition_path, recognition_model, feature_file):
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Face detector
        self.detector = SCRFD(model_file=detector_path)

        # Face recognizer
        self.recognizer = iresnet_inference(model_name=recognition_model, path=recognition_path, device=self.device)

        # Face Tracking
        self.tracker = self.load_config(tracking_path)
        self.tracking = BYTETracker(args=self.tracker, frame_rate=30)

        # Load precomputed face features and names
        self.images_names, self.images_embs = read_features(feature_path=feature_file)

        # Mapping of face IDs to names
        self.id_face_mapping = {}

        # Data mapping for tracking information
        self.data_mapping = {
            "raw_image": [],
            "tracking_ids": [],
            "detection_bboxes": [],
            "detection_landmarks": [],
            "tracking_bboxes": [],
        }

        # Camera object
        self.cap = cv2.VideoCapture(0)

        # FPS variables
        self.start_time = time.time_ns()
        self.frame_count = 0
        self.frame_id = 0
        self.fps = -1


        # Thread variables
        self.thread_track = None
        self.thread_recognize = None

    @staticmethod
    def load_config(file_name):
        """
        Load a YAML configuration file.

        Args:
            file_name (str): The path to the YAML configuration file.

        Returns:
            dict: The loaded configuration as a dictionary.
        """
        with open(file_name, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    @staticmethod
    def mapping_bbox(box1, box2):
        """
        Calculate the Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1 (tuple): The first bounding box (x_min, y_min, x_max, y_max).
            box2 (tuple): The second bounding box (x_min, y_min, x_max, y_max).

        Returns:
            float: The IoU score.
        """
        # Calculate the intersection area
        x_min_inter = max(box1[0], box2[0])
        y_min_inter = max(box1[1], box2[1])
        x_max_inter = min(box1[2], box2[2])
        y_max_inter = min(box1[3], box2[3])

        intersection_area = max(0, x_max_inter - x_min_inter + 1) * max(
            0, y_max_inter - y_min_inter + 1
        )

        # Calculate the area of each bounding box
        area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # Calculate the union area
        union_area = area_box1 + area_box2 - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area

        return iou

    @torch.no_grad()
    def get_feature(self, face_image):
        """
        Extract features from a face image.

        Args:
            face_image: The input face image.

        Returns:
            numpy.ndarray: The extracted features.
        """
        face_preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((112, 112)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Convert to RGB
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Preprocess image (BGR)
        face_image = face_preprocess(face_image).unsqueeze(0).to(self.device)

        # Inference to get feature
        emb_img_face = self.recognizer(face_image).cpu().numpy()

        # Convert to array
        images_emb = emb_img_face / np.linalg.norm(emb_img_face)

        return images_emb

    def recognition(self, face_image):
        """
        Recognize a face image.

        Args:
            face_image: The input face image.

        Returns:
            tuple: A tuple containing the recognition score and name.
        """
        # Get feature from face
        query_emb = self.get_feature(face_image)

        score, id_min = compare_encodings(query_emb, self.images_embs)
        name = self.images_names[id_min]
        score = score[0]

        return score, name

    def process_tracking(self, frame, detector, tracker, args, frame_id, fps):
        """
        Process tracking for a frame.

        Args:
            frame: The input frame.
            detector: The face detector.
            tracker: The object tracker.
            args (dict): Tracking configuration parameters.
            frame_id (int): The frame ID.
            fps (float): Frames per second.

        Returns:
            numpy.ndarray: The processed tracking image.
        """
        # Face detection and tracking
        outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)

        tracking_tlwhs = []
        tracking_ids = []
        tracking_scores = []
        tracking_bboxes = []

        if outputs is not None:
            online_targets = tracker.update( 
                outputs, [img_info["height"], img_info["width"]], (128, 128)
            )

            for i in range(len(online_targets)):
                t = online_targets[i]
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args["aspect_ratio_thresh"]
                if tlwh[2] * tlwh[3] > args["min_box_area"] and not vertical:
                    x1, y1, w, h = tlwh
                    tracking_bboxes.append([x1, y1, x1 + w, y1 + h])
                    tracking_tlwhs.append(tlwh)
                    tracking_ids.append(tid)
                    tracking_scores.append(t.score)

            tracking_image = plot_tracking(
                img_info["raw_img"],
                tracking_tlwhs,
                tracking_ids,
                names=self.id_face_mapping,
                frame_id=frame_id + 1,
                fps=fps,
            )
        else:
            tracking_image = img_info["raw_img"]

        self.data_mapping["raw_image"] = img_info["raw_img"]
        self.data_mapping["detection_bboxes"] = bboxes
        self.data_mapping["detection_landmarks"] = landmarks
        self.data_mapping["tracking_ids"] = tracking_ids
        self.data_mapping["tracking_bboxes"] = tracking_bboxes

        return tracking_image

    def start_camera(self):
        """Start the camera capture."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Couldn't read frame from camera.")
                break

            tracking_image = self.process_tracking(frame, self.detector, self.tracking, self.tracker, self.frame_id, self.fps)

            cv2.imshow("Face Recognition", tracking_image)

            # Check for user exit input
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

            # Update FPS
            self.update_fps()

        # Release the camera
        self.cap.release()
        cv2.destroyAllWindows()

    def update_fps(self):
        """Update the frames per second (fps) count."""
        self.frame_count += 1
        if self.frame_count >= 30:
            self.fps = 1e9 * self.frame_count / (time.time_ns() - self.start_time)
            self.frame_count = 0
            self.start_time = time.time_ns()

    def tracking_thread(self):
        """Face tracking in a separate thread."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Couldn't read frame from camera.")
                break

            tracking_image = self.process_tracking(frame, self.detector, self.tracking, self.tracker, self.frame_id, self.fps)

            cv2.imshow("Face Recognition", tracking_image)

            # Check for user exit input
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

            # Update FPS
            self.update_fps()

        # Release the camera
        self.cap.release()
        cv2.destroyAllWindows()

    def recognize_thread(self):
        """Face recognition in a separate thread."""
        while True:
            raw_image = self.data_mapping["raw_image"]
            detection_landmarks = self.data_mapping["detection_landmarks"]
            detection_bboxes = self.data_mapping["detection_bboxes"]
            tracking_ids = self.data_mapping["tracking_ids"]
            tracking_bboxes = self.data_mapping["tracking_bboxes"]

            for i in range(len(tracking_bboxes)):
                for j in range(len(detection_bboxes)):
                    mapping_score = self.mapping_bbox(box1=tracking_bboxes[i], box2=detection_bboxes[j])
                    if mapping_score > 0.9:
                        face_alignment = norm_crop(img=raw_image, landmark=detection_landmarks[j])

                        score, name = self.recognition(face_image=face_alignment)
                        if name is not None:
                            if score < 0.25:
                                caption = "UN_KNOWN"
                            else:
                                caption = f"{name}:{score:.2f}"

                        print("name: ", caption)

                        self.id_face_mapping[tracking_ids[i]] = caption

                        detection_bboxes = np.delete(detection_bboxes, j, axis=0)
                        detection_landmarks = np.delete(detection_landmarks, j, axis=0)

                        break

            # if tracking_bboxes == []:
            #     print("Waiting for a person...")    

    def start_threads(self):
        """Start the tracking and recognition threads."""
        self.thread_track = threading.Thread(target=self.tracking_thread)
        self.thread_track.start()

        self.thread_recognize = threading.Thread(target=self.recognize_thread)
        self.thread_recognize.start()

if __name__ == "__main__":
    main = FaceControl(
        detector_path="face_detector/scrfd/weights/scrfd_10g_bnkps.onnx",
        tracking_path="./face_tracking/config/config_tracking.yaml",
        recognition_path="face_recognition/arcface/weights/arcface_r100.pth",
        recognition_model="r100",
        feature_file="./datasets/face_features/feature"
    )
    main.start_threads()
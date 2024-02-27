import argparse
import os
import shutil

import time
import cv2

import numpy as np
import torch
from torchvision import transforms

from face_detector.scrfd.detector import SCRFD
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import read_features
class FaceDetector:
    def __init__(self, model_file, save_path, save_delay):
        # Initialize the face detector
        self.detector = SCRFD(model_file=model_file)
        self.cap = cv2.VideoCapture(0)  # Open the camera
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        self.video = cv2.VideoWriter("results/face-detection.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (self.frame_width, self.frame_height))
        self.start = time.time_ns()
        self.frame_count = 0
        self.fps = -1
        self.save_path = save_path  # Folder to save detected images
        self.image_count = 0  # Initialize image count
        self.last_save_time = 0.1  # Time of the last saved image
        self.save_delay = save_delay  # Minimum delay between saves in seconds
        

        # Create the directory if it does not exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def camera(self):
        camera_start_time = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process the captured frame for face detection
            self.process_img(frame)
            self.update_fps()

            # Save the frame to the video
            self.video.write(frame)

            # Show the result in a window
            cv2.imshow("Face Detection", frame)

            # NEW: Break the camera after 8 seconds
            if time.time() - camera_start_time > 8:
                print("Camera stopped after 8 seconds")
                break

            # Press 'Q' on the keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        self.cleanup()

    def process_img(self, frame):
        # Detect faces and landmarks
        bboxes, landmarks = self.detector.detect(image=frame)
        self.draw_detections(frame, bboxes, landmarks)

        # Save the image if faces are detected
        if len(bboxes) > 0:
            self.save_image(frame)

    def draw_detections(self, frame, bboxes, landmarks):
        h, w, _ = frame.shape
        tl = 1 or round(0.002 * (h + w) / 2) + 1  # Line and font thickness
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

        for i in range(len(bboxes)):
            x1, y1, x2, y2, score = bboxes[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)
            for id, key_point in enumerate(landmarks[i]):
                cv2.circle(frame, tuple(key_point), tl + 1, colors[id % len(colors)], -1)

    def update_fps(self):
        self.frame_count += 1
        if self.frame_count >= 30:
            end = time.time_ns()
            self.fps = 1e9 * self.frame_count / (end - self.start)
            self.frame_count = 0
            self.start = time.time_ns()
            print(f"FPS: {self.fps:.2f}")  # Or use cv2.putText to display it on the frame

    def save_image(self, frame):
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_delay:
            filename = f"{self.save_path}/image{self.image_count}.jpg"
            cv2.imwrite(filename, frame)
            print("save done")
            self.image_count += 1
            self.last_save_time = current_time  # Update the last save time

    def cleanup(self):
        self.video.release()
        self.cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

class Add_Persons:
    def __init__(self, backup_dir, add_persons_dir, faces_save_dir, features_path):

        # Check if CUDA is available and set the device accordingly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the face detector (Choose one of the detectors)
        self.detector = SCRFD(model_file="face_detector/scrfd/weights/scrfd_10g_bnkps.onnx")

        # Initialize the face recognizer
        self.recognizer = iresnet_inference(
            model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=self.device
        )

        self.backup_dir = backup_dir
        self.add_persons_dir = add_persons_dir
        self.faces_save_dir = faces_save_dir
        self.features_path = features_path

    @torch.no_grad()
    def get_feature(self, face_image):
        """
        Extract facial features from an image using the face recognition model.

        Args:
            face_image (numpy.ndarray): Input facial image.

        Returns:
            numpy.ndarray: Extracted facial features.
        """
        # Define a series of image preprocessing steps
        face_preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((112, 112)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Convert the image to RGB format
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Apply the defined preprocessing to the image
        face_image = face_preprocess(face_image).unsqueeze(0).to(self.device)

        # Use the model to obtain facial features
        emb_img_face = self.recognizer(face_image)[0].cpu().numpy()

        # Normalize the features
        images_emb = emb_img_face / np.linalg.norm(emb_img_face)
        return images_emb
    
    def add_persons(self):
        """
        Add a new person to the face recognition database.

        Args:
            backup_dir (str): Directory to save backup data.
            add_persons_dir (str): Directory containing images of the new person.
            faces_save_dir (str): Directory to save the extracted faces.
            features_path (str): Path to save face features.
        """
        # Initialize lists to store names and features of added images
        images_name = []
        images_emb = []
        # images_id = []

        # Read the folder with images of the new person, extract faces, and save them
        for name_person in os.listdir(self.add_persons_dir):
            person_image_path = os.path.join(self.add_persons_dir, name_person)

            # Create a directory to save the faces of the person
            person_face_path = os.path.join(self.faces_save_dir, name_person)
            os.makedirs(person_face_path, exist_ok=True)

            for image_name in os.listdir(person_image_path):
                if image_name.endswith(("png", "jpg", "jpeg")):
                    input_image = cv2.imread(os.path.join(person_image_path, image_name))

                    # Detect faces and landmarks using the face detector
                    bboxes, landmarks = self.detector.detect(image=input_image)

                    # Extract faces
                    for i in range(len(bboxes)):
                        # Get the number of files in the person's path
                        number_files = len(os.listdir(person_face_path))

                        # Get the location of the face
                        x1, y1, x2, y2, score = bboxes[i]

                        # Extract the face from the image
                        face_image = input_image[y1:y2, x1:x2]

                        # Path to save the face
                        path_save_face = os.path.join(person_face_path, f"{number_files}.jpg")

                        # Save the face to the database
                        cv2.imwrite(path_save_face, face_image)

                        # Extract features from the face
                        images_emb.append(self.get_feature(face_image=face_image))
                        images_name.append(name_person)

        # Check if no new person is found
        if images_emb == [] and images_name == []:
            print("No new person found!")
            return None

        # Convert lists to arrays
        images_emb = np.array(images_emb)
        images_name = np.array(images_name)

        # Read existing features if available
        features = read_features(self.features_path)

        if features is not None:
            # Unpack existing features
            old_images_name, old_images_emb = features

            # Combine new features with existing features
            images_name = np.hstack((old_images_name, images_name))
            images_emb = np.vstack((old_images_emb, images_emb))

            print("Update features!")

        # Save the combined features
        np.savez_compressed(self.features_path, images_name=images_name, images_emb=images_emb)

        # Move the data of the new person to the backup data directory
        for sub_dir in os.listdir(self.add_persons_dir):
            dir_to_move = os.path.join(self.add_persons_dir, sub_dir)
            shutil.move(dir_to_move, self.backup_dir, copy_function=shutil.copytree)

        print("Successfully added new person!")


if __name__ == "__main__":
    # name = input("User Name: ")
    code = input("User Code: ")

    fd = FaceDetector(
        model_file = "face_detector/scrfd/weights/scrfd_10g_bnkps.onnx",
        save_path = "/home/khuy/Recognition-System/datasets/new_persons/" + code,
        save_delay= 2
    )
    
    fd.camera()

    add_persons = Add_Persons(
            add_persons_dir="./datasets/new_persons",
            backup_dir="./datasets/backup",
            faces_save_dir="./datasets/data/",
            features_path="./datasets/face_features/feature"   
        )
    
    add_persons.add_persons()

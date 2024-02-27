import time
import cv2
import os
from face_detector.scrfd.detector import SCRFD

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
            self.image_count += 1
            self.last_save_time = current_time  # Update the last save time

    def cleanup(self):
        self.video.release()
        self.cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

if __name__ == "__main__":
    # name = input("User Name: ")
    code = input("User Code: ")

    fd = FaceDetector(
        model_file = "face_detector/scrfd/weights/scrfd_10g_bnkps.onnx",
        save_path = "/home/khuy/Recognition-System/datasets/new_persons/" + code,
        save_delay= 2
    )
    
    fd.camera()

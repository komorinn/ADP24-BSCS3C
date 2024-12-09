import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
import dlib
import numpy as np
import pickle
import os
from PIL import Image, ImageTk
import math
import sys

class FaceRecognitionGUI:
    def __init__(self, root):
        # Setup bundled resource paths
        if getattr(sys, 'frozen', False):
            self.bundle_dir = sys._MEIPASS
        else:
            self.bundle_dir = os.path.dirname(os.path.abspath(__file__))
            
        self.predictor_path = os.path.join(self.bundle_dir, 'shape_predictor_68_face_landmarks.dat')
        self.model_path = os.path.join(self.bundle_dir, 'dlib_face_recognition_resnet_model_v1.dat')
        self.face_db_path = os.path.join(self.bundle_dir, 'face_db.pkl')
        
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("800x600")
        
        # Initialize models
        self.detector, self.predictor, self.face_rec = self.initialize_models()
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Face Recognition System", font=('Helvetica', 20))
        title_label.pack(pady=20)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=20)
        
        ttk.Button(btn_frame, text="Train New Face", command=self.start_training, width=20).pack(pady=10)
        ttk.Button(btn_frame, text="Start Recognition", command=self.start_recognition, width=20).pack(pady=10)
        ttk.Button(btn_frame, text="Stop", command=self.stop_video, width=20).pack(pady=10)  # Added Stop button
        ttk.Button(btn_frame, text="Exit", command=root.quit, width=20).pack(pady=10)
        
        # Video frame
        self.video_frame = ttk.Label(main_frame)
        self.video_frame.pack(pady=20)
        
        self.cap = None
        self.is_training = False
        self.is_recognizing = False
        self.face_encodings = []
        self.capture_count = 0
        self.setup_key_bindings()

    def setup_key_bindings(self):
        self.root.bind('<c>', lambda e: self.handle_key_press('c'))
        self.root.bind('<q>', lambda e: self.handle_key_press('q'))

    def handle_key_press(self, key):
        if key == 'c' and self.is_training:
            self.capture_face(None)  # None because we'll get the frame from self.current_frame
        elif key == 'q':
            self.stop_video()

    def initialize_models(self):
        try:
            if not os.path.exists(self.predictor_path) or \
               not os.path.exists(self.model_path):
                messagebox.showerror("Error", "Required model files not found!")
                return None, None, None
            
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(self.predictor_path)
            face_rec = dlib.face_recognition_model_v1(self.model_path)
            return detector, predictor, face_rec
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize models: {str(e)}")
            return None, None, None

    def start_training(self):
        if not all([self.detector, self.predictor, self.face_rec]):
            messagebox.showerror("Error", "Models not properly initialized!")
            return
            
        self.name = simpledialog.askstring("Input", "Enter person's name:")
        if not self.name:
            return
            
        self.is_training = True
        self.is_recognizing = False
        self.capture_count = 0
        self.face_encodings = []
        self.start_video()
        messagebox.showinfo("Training", "Press 'c' to capture face (5 different angles needed)\nPress 'q' to quit")

    def start_recognition(self):
        if not all([self.detector, self.predictor, self.face_rec]):
            messagebox.showerror("Error", "Models not properly initialized!")
            return
            
        if not os.path.exists(self.face_db_path):
            messagebox.showerror("Error", "No trained faces found! Please train faces first.")
            return
            
        self.is_training = False
        self.is_recognizing = True
        self.start_video()

    def start_video(self):
        self.cap = cv2.VideoCapture(0)
        self.update_video()

    def update_video(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame  # Store current frame
                if self.is_training:
                    self.handle_training(frame)
                elif self.is_recognizing:
                    self.handle_recognition(frame)
                
                # Convert frame for display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.imgtk = imgtk
                self.video_frame.configure(image=imgtk)
                
                self.root.after(10, self.update_video)

    def draw_facial_features(self, frame, shape):
        # Draw facial landmarks
        for i in range(68):
            x, y = shape.part(i).x, shape.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Draw feature regions
        def connect_points(points, color=(0, 255, 0), closed=False):
            points = [(shape.part(i).x, shape.part(i).y) for i in points]
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], color, 1)
            if closed:
                cv2.line(frame, points[-1], points[0], color, 1)

        # Draw jaw line
        jaw_points = list(range(0, 17))
        connect_points(jaw_points)

        # Draw right eyebrow
        right_eyebrow = list(range(17, 22))
        connect_points(right_eyebrow)

        # Draw left eyebrow
        left_eyebrow = list(range(22, 27))
        connect_points(left_eyebrow)

        # Draw nose bridge
        nose_bridge = list(range(27, 31))
        connect_points(nose_bridge)

        # Draw lower nose
        lower_nose = list(range(30, 36))
        connect_points(lower_nose, closed=True)

        # Draw right eye
        right_eye = list(range(36, 42))
        connect_points(right_eye, closed=True)

        # Draw left eye
        left_eye = list(range(42, 48))
        connect_points(left_eye, closed=True)

        # Draw outer lip
        outer_lip = list(range(48, 60))
        connect_points(outer_lip, closed=True)

        # Draw inner lip
        inner_lip = list(range(60, 68))
        connect_points(inner_lip, closed=True)

    def handle_training(self, frame):
        faces = self.detector(frame)
        for face in faces:
            shape = self.predictor(frame, face)
            self.draw_facial_features(frame, shape)
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def handle_recognition(self, frame):
        try:
            with open(self.face_db_path, "rb") as f:
                database = pickle.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load database: {str(e)}")
            self.stop_video()
            return
            
        faces = self.detector(frame)
        for face in faces:
            shape = self.predictor(frame, face)
            self.draw_facial_features(frame, shape)
            face_descriptor = self.face_rec.compute_face_descriptor(frame, shape)
            
            name = self.find_match(face_descriptor, database)
            
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    def find_match(self, face_descriptor, database, threshold=0.5):
        min_distance = float("inf")
        best_match = "Unknown"
        for data in database:
            for encoding in data["encodings"]:
                distance = np.linalg.norm(np.array(face_descriptor) - np.array(encoding))
                if distance < min_distance:
                    min_distance = distance
                    best_match = data["name"]
        if min_distance < threshold:
            return best_match
        return "Unknown"

    def capture_face(self, frame=None):
        if frame is None:
            frame = self.current_frame
        if frame is None or self.capture_count >= 5:
            return
            
        faces = self.detector(frame)
        if len(faces) == 1:
            shape = self.predictor(frame, faces[0])
            face_descriptor = self.face_rec.compute_face_descriptor(frame, shape)
            face_descriptor = np.array(face_descriptor) / np.linalg.norm(face_descriptor)  # Normalize the descriptor
            self.face_encodings.append(face_descriptor)
            self.capture_count += 1
            messagebox.showinfo("Capture", f"Captured {self.capture_count}/5 faces")
            
            if self.capture_count == 5:
                self.save_training_data()
        else:
            messagebox.showwarning("Warning", "Please ensure only one face is visible")

    def save_training_data(self):
        data = {
            "name": self.name,
            "encodings": self.face_encodings
        }
        
        if os.path.exists(self.face_db_path):
            with open(self.face_db_path, "rb") as f:
                database = pickle.load(f)
        else:
            database = []
            
        database.append(data)
        
        with open(self.face_db_path, "wb") as f:
            pickle.dump(database, f)
            
        messagebox.showinfo("Success", f"Successfully trained face for {self.name}")
        self.stop_video()

    def stop_video(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_training = False
        self.is_recognizing = False
        self.video_frame.configure(image='')

    def __del__(self):
        if self.cap is not None:
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.mainloop()

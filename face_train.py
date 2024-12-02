import cv2
import dlib
import numpy as np
import pickle
import os

def train_face():
    try:
        print("Initializing face detector...")
        detector = dlib.get_frontal_face_detector()
        
        print("Loading shape predictor...")
        if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
            print("Error: shape_predictor_68_face_landmarks.dat not found!")
            print("Please run download_landmarks.py first")
            return
            
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        print("Loading face recognition model...")
        if not os.path.exists("dlib_face_recognition_resnet_model_v1.dat"):
            print("Error: dlib_face_recognition_resnet_model_v1.dat not found!")
            print("Please run download_landmarks.py first")
            return
            
        face_rec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

        print("Initializing webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam!")
            print("Please ensure your webcam is connected and not in use by another application")
            return
        
        # Get person's name
        name = input("Enter person's name: ")
        
        face_encodings = []
        print("Press 'c' to capture face (capture 5 different angles)")
        count = 0
        
        while count < 5:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Show frame
            cv2.imshow("Training", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                # Detect faces
                faces = detector(frame)
                if len(faces) == 1:  # Ensure only one face is detected
                    # Get face encoding
                    shape = predictor(frame, faces[0])
                    face_descriptor = face_rec.compute_face_descriptor(frame, shape)
                    face_encodings.append(face_descriptor)
                    count += 1
                    print(f"Captured {count}/5 faces")
                else:
                    print("Please ensure only one face is visible")
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if count == 5:
            # Save the encodings
            data = {
                "name": name,
                "encodings": face_encodings
            }
            
            # Create or load existing database
            if os.path.exists("face_db.pkl"):
                with open("face_db.pkl", "rb") as f:
                    database = pickle.load(f)
            else:
                database = []
                
            database.append(data)
            
            # Save updated database
            with open("face_db.pkl", "wb") as f:
                pickle.dump(database, f)
                
            print(f"Successfully trained face for {name}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    train_face()
    input("Press Enter to exit...")  # Keep window open to see error messages
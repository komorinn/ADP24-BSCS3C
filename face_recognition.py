import cv2
import dlib
import numpy as np
import pickle

# Initialize dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load the trained faces database
with open("face_db.pkl", "rb") as f:
    database = pickle.load(f)

def find_match(face_descriptor, threshold=0.6):
    for data in database:
        for encoding in data["encodings"]:
            # Calculate distance between face descriptors
            distance = np.linalg.norm(np.array(face_descriptor) - np.array(encoding))
            if distance < threshold:
                return data["name"]
    return "Unknown"

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    faces = detector(frame)

    # Process each face
    for face in faces:
        # Get facial landmarks and compute face descriptor
        shape = predictor(frame, face)
        face_descriptor = face_rec.compute_face_descriptor(frame, shape)
        
        # Find matching name
        name = find_match(face_descriptor)
        
        # Draw rectangle around face
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display name
        cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
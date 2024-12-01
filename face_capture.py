import cv2
from tkinter import *
from tkinter import ttk
from threading import Thread
import os

# Ensure directory exists 
os.makedirs("Dataset/Labang", exist_ok=True)

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Variables
img_id = 1
cap = cv2.VideoCapture(0)
capturing = False

class FaceCaptureUI:
    def __init__(self):
        self.window = Tk()
        self.window.title("Face Capture System")
        self.window.geometry("400x500")
        self.window.configure(bg='#f0f0f0')
        
        # Status variables
        self.status_var = StringVar(value="Ready")
        self.count_var = StringVar(value="Images: 0")
        
        self.create_widgets()
        
    def create_widgets(self):
        # Title Frame
        title_frame = Frame(self.window, bg='#f0f0f0')
        title_frame.pack(pady=20)
        
        Label(title_frame, 
              text="Face Capture System",
              font=("Helvetica", 24, "bold"),
              bg='#f0f0f0').pack()
        
        # Status Frame
        status_frame = Frame(self.window, bg='#f0f0f0')
        status_frame.pack(pady=20, fill=X, padx=20)
        
        # Status indicator
        self.status_indicator = Label(status_frame,
                                    textvariable=self.status_var,
                                    font=("Helvetica", 12),
                                    bg='#e0e0e0',
                                    relief=RIDGE,
                                    padx=10,
                                    pady=5)
        self.status_indicator.pack(fill=X)
        
        # Counter
        Label(status_frame,
              textvariable=self.count_var,
              font=("Helvetica", 12),
              bg='#f0f0f0').pack(pady=10)
        
        # Buttons Frame
        button_frame = Frame(self.window, bg='#f0f0f0')
        button_frame.pack(pady=20)
        
        # Start Button
        self.start_button = ttk.Button(button_frame,
                                     text="Start Capture",
                                     style='Green.TButton',
                                     command=self.start_capture)
        self.start_button.pack(pady=10)
        
        # Stop Button
        self.stop_button = ttk.Button(button_frame,
                                    text="Stop Capture",
                                    style='Red.TButton',
                                    command=self.stop_capture,
                                    state=DISABLED)
        self.stop_button.pack(pady=10)
        
        # Configure styles
        style = ttk.Style()
        style.configure('Green.TButton',
                       font=('Helvetica', 12),
                       padding=10)
        style.configure('Red.TButton',
                       font=('Helvetica', 12),
                       padding=10)
        
    def update_status(self, status, count):
        self.status_var.set(status)
        self.count_var.set(f"Images: {count}")
        
    def start_capture(self):
        global capturing
        capturing = True
        self.start_button.configure(state=DISABLED)
        self.stop_button.configure(state=NORMAL)
        self.update_status("Capturing...", img_id)
        startCapture()
        
    def stop_capture(self):
        global capturing
        capturing = False
        self.start_button.configure(state=NORMAL)
        self.stop_button.configure(state=DISABLED)
        self.update_status("Stopped", img_id)
        stopCapture()

# function to capture images
def captureImgSample(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = faceCascade.detectMultiScale(gray_img, 1.1, 10)
    coordinate = []
    check = 0

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, "Face", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
        coordinate = [x, y, w, h]

    print(coordinate)
    if len(coordinate) == 4:
        roi_img = img[coordinate[1]:coordinate[1] + coordinate[3], coordinate[0]:coordinate[0] + coordinate[2]]
        user_id = 1
        check = 1
        cv2.imwrite("Dataset/Labang/Image." + str(user_id) + "." + str(img_id) + ".jpg", roi_img)
        app.update_status("Face Captured", img_id)
    return check

# function for capturing
def startCapture():
    global capturing, img_id
    capturing = True
    img_id = 1

    # function for threading and capturing images
    def capture_loop():
        global img_id
        while capturing:
            success, img = cap.read()
            if not success:
                print("Error: Failed to capture image.")
                break
            check = captureImgSample(img)

            print(img_id)
            if check == 1:
                img_id += 1

            cv2.imshow("Capturing Dataset", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture_thread = Thread(target=capture_loop)  # Added threading to avoid blocking the main thread / able to still use start and stpo
    capture_thread.start()  # Start the capture loop in a separate thread

# stopping function
def stopCapture():
    global capturing
    capturing = False
    cap.release()
    cv2.destroyAllWindows()

def cleanup():
    global capturing
    capturing = False
    cap.release()
    cv2.destroyAllWindows()
    app.window.quit()

# Create and run UI
if __name__ == "__main__":
    app = FaceCaptureUI()
    app.window.mainloop()
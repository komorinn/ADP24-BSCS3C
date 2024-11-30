import cv2
from tkinter import *
from threading import Thread  # Added import for threading

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# variable
img_id = 1
cap = cv2.VideoCapture(0)
capturing = False  # this is for stopping the capturing / end capture.

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

# Tkinter Window
window = Tk()
window.title("Start Capturing Dataset.")
window.geometry("300x200")

label = Label(window,
              text="Ready to Capture?",
              font=("Arial", 18, "bold",))
label.pack(pady=20)

start_button = Button(window,
                      text="Start Capture",
                      font=("Arial", 14),
                      bg="green",
                      fg="white",
                      activeforeground="white",
                      activebackground="green",
                      padx=10,
                      pady=5,
                      command=startCapture)
start_button.pack()

stop_button = Button(window,
                     text="Stop Capture",
                     font=("Arial", 14),
                     bg="red",
                     fg="white",
                     activeforeground="white",
                     activebackground="red",
                     padx=10,
                     pady=5,
                     command=stopCapture)
stop_button.pack()

window.mainloop()
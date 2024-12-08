
try:
    import cv2
    print("OpenCV successfully imported!")
    print("OpenCV version:", cv2.__version__)
except ImportError as e:
    print("Error importing OpenCV:", str(e))
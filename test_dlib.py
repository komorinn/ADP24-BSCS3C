
try:
    import dlib
    print("dlib successfully imported!")
    print("dlib version:", dlib.__version__)
except ImportError as e:
    print("Error importing dlib:", str(e))

try:
    import requests
    print("requests library successfully imported!")
    print("requests version:", requests.__version__)
except ImportError as e:
    print("Error importing requests:", str(e))
import requests
import bz2
import os

def download_and_extract(url, filename):
    print(f"Downloading {filename}...")
    response = requests.get(url)
    
    # Save the compressed file
    compressed_file = f"{filename}.bz2"
    with open(compressed_file, "wb") as f:
        f.write(response.content)
    
    print(f"Extracting {filename}...")
    try:
        # Properly decompress bz2 file
        with bz2.open(compressed_file, "rb") as source, open(filename, "wb") as dest:
            dest.write(source.read())
        print(f"Successfully extracted {filename}")
    except Exception as e:
        print(f"Error extracting {filename}: {str(e)}")
        return False
    finally:
        # Clean up compressed file
        if os.path.exists(compressed_file):
            os.remove(compressed_file)
    return True

def download_models():
    # Delete existing files if they exist
    files_to_check = [
        "shape_predictor_68_face_landmarks.dat",
        "dlib_face_recognition_resnet_model_v1.dat"
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Removed existing {file}")
            except Exception as e:
                print(f"Error removing {file}: {str(e)}")

    # URLs for both model files
    models = {
        "shape_predictor_68_face_landmarks.dat": 
            "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
        "dlib_face_recognition_resnet_model_v1.dat":
            "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
    }
    
    success = True
    for filename, url in models.items():
        if not download_and_extract(url, filename):
            success = False
            break
    
    if success:
        print("All models successfully downloaded and extracted!")
    else:
        print("Error occurred during download/extraction")

if __name__ == "__main__":
    download_models()
    input("Press Enter to exit...")
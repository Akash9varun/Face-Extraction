import cv2
import zipfile
import os
import streamlit as st
import shutil
import tempfile

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Streamlit app
def app():
    st.title('Face Extraction from Video')

    # Display a file uploader for the video file
    f = st.file_uploader("Upload file")
    if f is not None:
        # Save the uploaded file to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(f.read())

        # Extract faces from the video file when the user clicks the button
        if st.button('Extract Faces'):
            # Open the video file
            video_capture = cv2.VideoCapture(tfile.name)

            # Initialize a list to store the extracted faces
            faces = []

            # Loop through each frame in the video
            while True:
                # Read a frame from the video
                ret, frame = video_capture.read()
                if not ret:
                    break

                # Convert the frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the grayscale frame
                detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                # Extract each face from the frame
                for (x, y, w, h) in detected_faces:
                    face = frame[y:y+h, x:x+w]
                    faces.append(face)

            # Release the video capture object
            video_capture.release()

            # Create a directory to store the extracted faces
            directory = 'faces'
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Loop through each face and save it as an image file in the directory
            for i, face in enumerate(faces):
                filename = os.path.join(directory, f'face_{i}.jpg')
                cv2.imwrite(filename, face)

            # Create a zip file to store the extracted faces
            with zipfile.ZipFile('faces.zip', 'w') as zip_file:
                # Loop through each face image file and add it to the zip file
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        filename = os.path.join(root, file)
                        zip_file.write(filename)

            # Display a link to download the zip file
            with open('faces.zip', 'rb') as f:
                bytes = f.read()
                st.download_button('Download Faces', data=bytes, file_name='faces.zip')

            # Delete the temporary file
            os.unlink(tfile.name)

            # Delete the directory and its contents
            shutil.rmtree(directory)
if __name__ == '__main__':
    app()



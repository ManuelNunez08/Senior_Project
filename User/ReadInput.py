import cv2
import os
from moviepy.editor import VideoFileClip

# to load audio model 
import torch

# to load visual model 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
import tensorflow as tf
import numpy as np

# for directory reading 
import glob




def extract_faces_and_audio(video_path, output_dir):
    # Load face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract audio from video and save
    video_clip = VideoFileClip(video_path)
    audio_path = os.path.join(output_dir, 'audio.wav')
    video_clip.audio.write_audiofile(audio_path)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    screenshot_count = 1

    while True:
        # Read frame from video
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if no frames are left

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # If faces are detected, save them as images
        for (x, y, w, h) in faces:
            if frame_count%50 == 0:
                face_img = frame[y:y+h, x:x+w]
                cv2.imwrite(os.path.join(output_dir, f'face_{screenshot_count}.jpg'), face_img)
                screenshot_count += 1

        frame_count += 1

    cap.release()




# We first rea din the input and split it into two contexts. The data, nowe ready for classification, is situated in the split_input folder
video_path = 'input/input.mov'
output_dir = 'split_input'
extract_faces_and_audio(video_path, output_dir)

# We now load the visual model 
visual_model = tf.keras.models.load_model('models/firstmodel.keras')

# the visual model evaluates each screenshot below 
visual_predictions = []
path_to_images = 'split_input/visual'
for file_name in os.listdir(path_to_images):
    img_path = os.path.join(path_to_images, file_name)
    img = image.load_img(img_path, target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array_expanded_dims)
    prediction = visual_model.predict(img_preprocessed)
    predicted_class = np.argmax(prediction, axis=1)
    visual_predictions.append(predicted_class)




import cv2
import os
from moviepy.editor import VideoFileClip

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

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


# AUDIO MODEL
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Adjust the input size accordingly
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)

    def forward(self, spec_data):
        x = self.pool(F.relu(self.conv1(spec_data)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = self.adaptive_pool(x)

        # Flatten the output for the dense layers
        x = x.view(-1, 64 * 5 * 5)

        # Pass the flattened output through dense layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

model = AudioCNN()
model.load_state_dict(torch.load('../Models/Audio_model/audio_model_path.pth'))

def full_pipeline(file_path):
    # read in the data from the file its in
    y, sr = librosa.load(file_path)

    # preprocess this read in data
    ms = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    y = librosa.amplitude_to_db(ms, ref=np.max)
    y = torch.tensor(y)
    y = y.unsqueeze(0)

# obtain model output and record it
    with torch.no_grad():
        outputs = model(y)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)

    return outputs


# We now load the audio model
audio_path = 'split_input/audio/audio.wav'
audio_results = full_pipeline(audio_path)
print(audio_results)



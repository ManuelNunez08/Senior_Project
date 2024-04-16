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

# for turing predictions to json 
import json




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




# We first rea in the input and split it into two contexts. The data, now ready for classification, is situated in the split_input folder
video_path = 'input/input.mov'
output_dir = 'split_input'
extract_faces_and_audio(video_path, output_dir)



# EMPLOY VISUAL MODEL 

# We now load the visual model
visual_model = tf.keras.models.load_model('/Users/manuelnunezmartinez/Documents/UF/Spring24/Senior Project/Senior_Project/User/models/firstmodel.keras')
# the visual model evaluates each screenshot below
visual_predictions = []
path_to_images = 'split_input/visual'

# Initialize a list to store the sum of predictions for each class
sum_predictions = None
num_images = 0
# Iterate through each image and take average of each classification
for file_name in os.listdir(path_to_images):
    if file_name.endswith('.jpg'):  # Make sure to process .jpg files only
        img_path = os.path.join(path_to_images, file_name)
        img = image.load_img(img_path, target_size=(48, 48))
        img_array = image.img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array_expanded_dims)
        prediction = visual_model.predict(img_preprocessed)
        # sum predictions 
        if sum_predictions is None:
            sum_predictions = prediction
        else:
            sum_predictions += prediction
        num_images += 1

# Calculate the average predictions for each class
visual_predictions = sum_predictions / num_images if num_images else []
# If you need the result as a regular Python list
visual_predictions = visual_predictions.tolist()[0] 





# EMPLOY AUDIO MODEL 

# Define class for model loading
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


# Load the model 
model = AudioCNN()
model.load_state_dict(torch.load('/Users/manuelnunezmartinez/Documents/UF/Spring24/Senior Project/Senior_Project/User/models/audio_model_path.pth'))

# Function to evaluate input given path 
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
audio_path = '/Users/manuelnunezmartinez/Documents/UF/Spring24/Senior Project/Senior_Project/User/split_input/audio/audio.wav'
audio_predictions = full_pipeline(audio_path)
audio_predictions = audio_predictions.tolist()





# CONSTRUCT PROPER JSON OBJECT TO SEND TO BACKEND

# Contempt and Calmness are not considered as they arent present in both data sets 

results_dict = {}

# assign emotions to corresponding list index for both models 
audio_dict = {'Angry': audio_predictions[0], 'Disgust': audio_predictions[2], 
            'Fearful':audio_predictions[3], 'Happy': audio_predictions[4], 'Neutral':audio_predictions[5], 
            'Sad':audio_predictions[6], 'Surprise': audio_predictions[7]}

visual_dict = {'Angry': visual_predictions[0], 'Disgust': visual_predictions[2], 
            'Fearful':visual_predictions[3], 'Happy': visual_predictions[4], 'Neutral':visual_predictions[5], 
            'Sad':visual_predictions[6], 'Surprise': visual_predictions[7]}

# Find complex emotions within each complex 
visual_complex_emotions = {
                        'Dissaproval':(visual_dict['Sad'] + visual_dict['Surprise'])/2,
                        'Remorse': (visual_dict['Sad'] + visual_dict['Disgust'])/2,
                        'Contempt': (visual_dict['Disgust'] + visual_dict['Anger'])/2,
                        'Awe': (visual_dict['Fear'] + visual_dict['Surprise'])/2,
                        'Excitement': (visual_dict['Happy'] + visual_dict['Surprise'])/2
                        }
audio_complex_emotions = {
                        'Dissaproval':(audio_dict['Sad'] + audio_dict['Surprise'])/2,
                        'Remorse': (audio_dict['Sad'] + audio_dict['Disgust'])/2,
                        'Contempt': (audio_dict['Disgust'] + audio_dict['Anger'])/2,
                        'Awe': (audio_dict['Fear'] + audio_dict['Surprise'])/2,
                        'Excitement': (audio_dict['Happy'] + audio_dict['Surprise'])/2
                        }


# Dictionary for complex emotions obtained py combining contexts (We take the combination which is greatest)
complex_emotions_combined_contexts = {
                        'Dissaproval': max((audio_dict['Sad'] + visual_dict['Surprise'])/2,  (audio_dict['Surprise'] + visual_dict['Sad'])/2),
                        'Remorse': max((audio_dict['Sad'] + visual_dict['Disgust'])/2,  (audio_dict['Disgust'] + visual_dict['Sad'])/2),
                        'Contempt': max((audio_dict['Disgust'] + visual_dict['Anger'])/2,  (audio_dict['Anger'] + visual_dict['Disgust'])/2),
                        'Awe': max((audio_dict['Fear'] + visual_dict['Surprise'])/2,  (audio_dict['Surprise'] + visual_dict['Fear'])/2),
                        'Excitement': max((audio_dict['Happy'] + visual_dict['Surprise'])/2,  (audio_dict['Surprise'] + visual_dict['Happy'])/2)
                        }


results_dict['Visual_Predictions'] = visual_dict
results_dict['Audio_Predictions'] = audio_dict
results_dict ['Visual_Complex'] = visual_complex_emotions
results_dict ['Audio_Complex'] = audio_complex_emotions
results_dict ['Combined_Complex'] = complex_emotions_combined_contexts


# write out json file 
json_path = 'results.json'

# Writing the JSON data to a file
with open(json_path, 'w') as json_file:
    json.dump(results_dict, json_file, indent=4)  

from moviepy.editor import VideoFileClip
import os

# We first read in the input and split it into two contexts. The data, now ready for classification, is situated in the split_input folder
video_path = '../FrontBack/saved-videos/converted_video.mp4'
output_dir = 'split_input'


import cv2

def get_video_frame_count(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return -1

    # Get the total number of frames in the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Clean up: release the video capture object
    cap.release()
    
    return frame_count

def check_video_frame_rate(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

frame_count = get_video_frame_count(video_path)
print(f"The video has {frame_count} frames.")

fps = check_video_frame_rate(video_path)
print(f"The video frame rate is: {fps} FPS.")
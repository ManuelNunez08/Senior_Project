import cv2
import os
import time

# Load face detection 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)  


output_directory = 'new_input'
os.makedirs(output_directory, exist_ok=True)

# Set video recording parameters
fps = 30  
duration = 5  
start_time = time.time()

# Initialize frame count and screenshot counter
frame_count = 0
screenshot_count = 0  # Start with 0, increment before save

# Loop until duration is reached or 10 screenshots have been taken
while time.time() - start_time < duration and screenshot_count < 10:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # If faces are detected and less than 10 screenshots taken, take a screenshot
    if len(faces) > 0 and screenshot_count < 10:
        if frame_count % 10 == 0:
            # Save screenshots of detected faces
            for (x, y, w, h) in faces:
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Crop face region
                face_region = frame[y:y+h, x:x+w]
                screenshot_count += 1  # Increment before saving
                # Save screenshot
                cv2.imwrite(os.path.join(output_directory, f'face_{screenshot_count}.jpg'), face_region)
                if screenshot_count >= 10:  # Check after increment
                    break  # Exit for-loop once 10 images are saved
    
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    # Increment frame count
    frame_count += 1
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename


app = Flask(__name__)
VIDEO_FOLDER = 'saved-videos'

# Create a directory for saved-videos if it doesn't exist
if not os.path.exists(VIDEO_FOLDER):
    os.makedirs(VIDEO_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Define a secure filename
    filename = secure_filename(video.filename)
    # Save video to the saved-videos folder
    video.save(os.path.join(VIDEO_FOLDER, filename))
    
    # Process your video here with your ML model
    
    return jsonify({'message': 'Video received and saved'}), 200

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template, render_template_string
import os
from werkzeug.utils import secure_filename
import plotly.graph_objects as go
import plotly.io as pio
import json 


app = Flask(__name__)
VIDEO_FOLDER = 'saved-videos'

# Get the directory where the script is located
script_dir = os.path.dirname(__file__)

RESULTS_PATH = os.path.join(script_dir, '../User/results.json')



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

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/visualize')
def visualize():
    # Open and read the JSON file
    with open(RESULTS_PATH) as file:
        data_dict = json.load(file)

    plots_div = []

    # Bar charts for the first two
    for key in ['Visual_Predictions', 'Audio_Predictions']:
        fig = go.Figure([go.Bar(x=list(data_dict[key].keys()), y=list(data_dict[key].values()))])
        fig.update_layout(title_text=key.replace('_', ' '))
        plots_div.append(pio.to_html(fig, full_html=False))

    # Pie charts for the remaining three
    for key in ['Visual_Complex', 'Audio_Complex', 'Combined_Complex']:
        fig = go.Figure([go.Pie(labels=list(data_dict[key].keys()), values=list(data_dict[key].values()))])
        fig.update_layout(title_text=key.replace('_', ' '))
        plots_div.append(pio.to_html(fig, full_html=False))

    # Render all plots in an HTML template
    html_template = """
<!DOCTYPE html>
<html>
    <head>
        <title>Data Visualization</title>
    </head>
    <body>
        <h2>Data Visualization</h2>
        {}
    </body>
</html>
""".format("".join(plots_div))  # Insert plots into HTML

    return render_template_string(html_template)



if __name__ == '__main__':
    app.run(debug=True)

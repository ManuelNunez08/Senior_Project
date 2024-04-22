from flask import Flask, request, jsonify, render_template, render_template_string, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import json 
import subprocess
import sys
import threading
from interpretEmotions import get_interpretation
import time

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*')

VIDEO_FOLDER = 'saved-videos'

# Get the directory where the script is located
script_dir = os.path.dirname(__file__)

RESULTS_PATH = os.path.join(script_dir, 'results.json')

READ_INPUT_PATH = os.path.join(script_dir, '../User/ReadInput.py')

VIDEO_FILE = os.path.join(VIDEO_FOLDER, 'video.mp4')


# Create a directory for saved-videos if it doesn't exist
if not os.path.exists(VIDEO_FOLDER):
    os.makedirs(VIDEO_FOLDER)

@app.route('/record-video')
def record_video():
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

    # Convert the video to a compatible format
    output_path = os.path.join(VIDEO_FOLDER, 'converted_video.mp4')

    command = [
        'ffmpeg', '-y',
        '-i', VIDEO_FILE,
        '-r', '30',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'aac', '-b:a', '192k',
        output_path
    ]
    try: 
        try:
            subprocess.run(command, check=True)
            print("Conversion successful")
            # After conversion, run the ReadInput.py script
            subprocess.run([sys.executable, READ_INPUT_PATH], check=True)
            print("Processing successful")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")

    
        return jsonify({"success": True})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    


@app.route('/processing-complete', methods=['POST'])
def processing_complete():
    data = request.json
    print("Processing complete with data:", data)
    socketio.emit('processing_done', {'message': 'Processing complete'})
    return jsonify({'status': 'success', 'message': 'Notified frontend'}), 200

@app.route('/home')
def home():
    # Define the path to your JSON file
    EXAMPLE_RESULTS_PATH = os.path.join(app.root_path, 'static', 'home_page/Example_results.json')
    
    # Open and read the JSON file
    with open(EXAMPLE_RESULTS_PATH) as file:
        data_dict = json.load(file)

    # Initialize a dictionary to hold our plotly divs
    plots_div = {}
    color_palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    # Customize the plot size here
    plot_width = 600
    plot_height = 400

    # Bar charts for the visual and audio predictions
    for key in ['Visual_Predictions', 'Audio_Predictions']:
        fig = go.Figure([go.Bar(x=list(data_dict[key].keys()), y=list(data_dict[key].values()), marker_color=color_palette)])
        fig.update_layout(title_text=key.replace('_', ' '), title_x=0.5, width=plot_width/1.5, height=plot_height)
        plots_div[key] = pio.to_html(fig, full_html=False, include_plotlyjs=False, config={'staticPlot': True})

    # Pie charts for the complex emotions
    for key in ['Visual_Complex', 'Audio_Complex', 'Combined_Complex']:
        fig = go.Figure([go.Pie(labels=list(data_dict[key].keys()), values=list(data_dict[key].values()))])
        fig.update_layout(title_text=key.replace('_', ' '), title_x=0.5, legend=dict(x=0), width=plot_width, height=plot_height)
        plots_div[key] = pio.to_html(fig, full_html=False, include_plotlyjs=False, config={'staticPlot': True})

    # Render the homepage with the plots
    return render_template('home.html', plots=plots_div)


@app.route('/visualize')
def visualize():
    # Open and read the JSON file
    with open(RESULTS_PATH) as file:
        data_dict = json.load(file)
        
    color_palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    
    fig_bar = make_subplots(rows=1, cols=2, subplot_titles=('Visual Predictions', 'Audio Predictions'))
    bar_charts = ['Visual_Predictions', 'Audio_Predictions']
    for i, key in enumerate(bar_charts, start=1):
        fig_bar.add_trace(
            go.Bar(x=list(data_dict[key].keys()), y=list(data_dict[key].values()), marker_color=color_palette, 
                   name=key.replace('_', ' ')),
            row=1, col=i
        )
    fig_bar.update_layout(height=400, width=800, showlegend=False, title_text="Bar Chart Visualizations")

    # Second subplot with pie charts
    fig_pie = make_subplots(rows=1, 
                            cols=3, 
                            subplot_titles=('Visual Complexity', 'Audio Complexity', 'Combined Complexity'),
                            specs=[[{"type":"pie"}, {"type":"pie"}, {"type":"pie"}]])
    pie_charts = ['Visual_Complex', 'Audio_Complex', 'Combined_Complex']
    for i, key in enumerate(pie_charts, start=1):
        fig_pie.add_trace(
            go.Pie(labels=list(data_dict[key].keys()), values=list(data_dict[key].values()), 
                   name=key.replace('_', ' ')),
            row=1, col=i
        )
    fig_pie.update_layout(height=400, width=800, title_text="Pie Chart Visualizations")



    # Convert both subplot figures to HTML
    plot_html_bar = pio.to_html(fig_bar, full_html=False, include_plotlyjs='cdn', config={'staticPlot': True})
    plot_html_pie = pio.to_html(fig_pie, full_html=False, include_plotlyjs=False, config={'staticPlot': True}) 

    # obtain f strings for bar charts and complex emotions. 
    visual_message = get_interpretation(data_dict['Visual_Predictions'], 'visual context')
    audio_message = get_interpretation(data_dict['Audio_Predictions'], 'auditory context')
    visual_complex_message = get_interpretation(data_dict['Visual_Complex'], 'complex visual context')
    audio_complex_message = get_interpretation(data_dict['Audio_Complex'], 'complex auditory context')
    mixed_complex_message = get_interpretation(data_dict['Combined_Complex'], 'combined complex context')



    return render_template('visualize.html', 
                            plot_html_bar=plot_html_bar, 
                            plot_html_pie=plot_html_pie,
                            visual_message=visual_message,
                            audio_message=audio_message,
                            visual_complex_message=visual_complex_message,
                            audio_complex_message=audio_complex_message,
                            mixed_complex_message=mixed_complex_message
                            )

@app.route('/saved-videos/<filename>')
def serve_video(filename):
    video_directory = os.path.join(app.root_path, VIDEO_FOLDER)
    return send_from_directory(video_directory, filename)



# @app.route('/start-recording', methods=['POST'])
# def start_recording():
#     try:
#         # Placeholder for your Python code that records video and processes files
#         # Simulating a delay for demonstration purposes
#         # Simulate time taken to record and process the video
        
#         return jsonify({"success": True})
#     except Exception as e:
#         return jsonify({"success": False, "error": str(e)})


if __name__ == '__main__':
    socketio.run(app, allow_unsafe_werkzeug=True)

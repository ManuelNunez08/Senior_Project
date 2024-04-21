from flask import Flask, request, jsonify, render_template, render_template_string
import os
from werkzeug.utils import secure_filename
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
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

@app.route('/visualize')
def visualize():
    # Open and read the JSON file
    with open(RESULTS_PATH) as file:
        data_dict = json.load(file)
        
    fig_bar = make_subplots(rows=1, cols=2, subplot_titles=('Visual Predictions', 'Audio Predictions'))
    bar_charts = ['Visual_Predictions', 'Audio_Predictions']
    for i, key in enumerate(bar_charts, start=1):
        fig_bar.add_trace(
            go.Bar(x=list(data_dict[key].keys()), y=list(data_dict[key].values()), 
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
    plot_html_bar = pio.to_html(fig_bar, full_html=False, include_plotlyjs='cdn')
    plot_html_pie = pio.to_html(fig_pie, full_html=False, include_plotlyjs=False)  # No need to include Plotly.js again

    # Render both plots in a single HTML template, stacked vertically
    html_template = """
        <!DOCTYPE html>
        <html>
            <head>
                <title>Data Visualization</title>
            </head>
            <body>
                {plot_html_bar}
                {plot_html_pie}
            </body>
        </html>
        """.format(plot_html_bar=plot_html_bar, plot_html_pie=plot_html_pie)

    return render_template_string(html_template)



if __name__ == '__main__':
    app.run(debug=True)

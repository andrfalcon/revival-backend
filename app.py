from flask import Flask, request, jsonify, Response, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import numpy as np
from model import Model
from helpers import save_video_clip, most_repeated_clusters
app = Flask(__name__)
CORS(app)

# Initialize project instance
my_model = Model()

# Configure video upload folder
VIDEOS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'videos')
os.makedirs(VIDEOS_FOLDER, exist_ok=True)
CLUSTER_CLIPS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cluster_clips')
os.makedirs(CLUSTER_CLIPS_FOLDER, exist_ok=True)

@app.route('/videos/<cluster_id>')
def serve_video(cluster_id):
    if os.path.exists(os.path.join(CLUSTER_CLIPS_FOLDER, f'{cluster_id}.mp4')):
        return send_from_directory(CLUSTER_CLIPS_FOLDER, f'{cluster_id}.mp4')
    else:
        cluster_clip_data = my_model.get_cluster_clip(int(cluster_id))
        cluster_clip_video_path = cluster_clip_data['video_path']
        cluster_clip_time_start = cluster_clip_data['time_start']
        cluster_clip_time_end = cluster_clip_data['time_end']
        save_video_clip(
            cluster_clip_video_path,
            cluster_clip_time_start,
            cluster_clip_time_end,
            CLUSTER_CLIPS_FOLDER,
            f'{cluster_id}.mp4'
        )
        return send_from_directory(CLUSTER_CLIPS_FOLDER, f'{cluster_id}.mp4')

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    saved_files = []

    for file in files:
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(VIDEOS_FOLDER, filename)
            file.save(file_path)
            saved_files.append(filename)

    if saved_files:
        return jsonify({'message': f'Successfully uploaded {len(saved_files)} files', 'files': saved_files}), 200
    else:
        return jsonify({'error': 'No valid files were uploaded'}), 400


@app.route('/analyze', methods=['GET'])
def analyze_video():
    my_model.run()
    cluster_data = my_model.get_cluster_data()
    print(cluster_data)
    # Only return the 30 most repeated tasks back to the frontend
    # TODO: Implement frontend feature where this val can be inputted by user
    cluster_data = most_repeated_clusters(cluster_data, 30)
    return jsonify(cluster_data)

if __name__ == '__main__':
    app.run(debug=True)





# @app.route('/video/<filename>')
# def serve_video(filename):
#     """
#     Serve video files from the videos directory
#     """
#     try:
#         return send_from_directory(VIDEOS_FOLDER, filename)
#     except FileNotFoundError:
#         return {"error": "Video not found"}, 404
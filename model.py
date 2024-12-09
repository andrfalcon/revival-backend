import os
import cv2
import base64
import replicate
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from dotenv import load_dotenv
load_dotenv()
os.environ.get("REPLICATE_API_TOKEN")

class Model():
    video_folder = './videos/'
    frame_folder = './frames/'
    interval = 2

    def __init__(self):
        self.count = 0
        self.data = {}
        self.cluster_data = {}
        self.embeddings = []

    def run(self):
        video_paths = []

        for video_name in os.listdir(Model.video_folder):
            video_paths.append(os.path.join(Model.video_folder, video_name))

        for video in video_paths:
            self.extract_embed_delete(video)

        self.hierarchical_cluster()

    def extract_embed_delete(self, video_path):
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            print("Error: Could not open video.")
            return

        fps = video.get(cv2.CAP_PROP_FPS)
        interval_frames = int(fps * Model.interval)

        frame_count = 0
        saved_frame_count = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break

            if frame_count % interval_frames == 0:
                try:
                    frame_filename = os.path.join(Model.frame_folder, f"{self.count}.jpg")
                    cv2.imwrite(frame_filename, frame)

                    self.embed(frame_filename, video_path)
                    self.delete(frame_filename)
                except:
                    print("One frame unable to be processed. This frame will not be included in the final analysis.")
                else:
                    print(f"Saved {frame_filename}")
                    saved_frame_count += 1
                    self.count += 1

            frame_count += 1

        video.release()
        print("Done!")

    def embed(self, frame_filename, video_path):
        with open(frame_filename, 'rb') as file:
            data = base64.b64encode(file.read()).decode('utf-8')

        input = f"data:application/octet-stream;base64,{data}"
        input = {
            "input": input
        }

        output = replicate.run(
            "daanelson/imagebind:0383f62e173dc821ec52663ed22a076d9c970549c209666ac3db181618b7a304",
            input=input
        )

        self.data[self.count] = {
            "embedding": output,
            "time_start": Model.interval * self.count,
            "time_end": (Model.interval * self.count) + Model.interval,
            "video_path": video_path,
            "cluster_id": None
        }

        self.embeddings.append(output)

    def delete(self, frame_filename):
        os.remove(frame_filename)

    def hierarchical_cluster(self):
        # Assuming X is your 2D array of vectors
        X = np.array(self.embeddings)
        dist_matrix = pdist(X)
        linkage_matrix = linkage(dist_matrix, method='ward')
        max_d = 0.5  # maximum distance for forming clusters
        clusters = fcluster(linkage_matrix, max_d, criterion='distance')
        print("CLUSTERS: ", clusters)

        # Create a dictionary to store vectors for each cluster
        cluster_dict = {}
        for i, cluster in enumerate(clusters):
            if cluster not in cluster_dict:
                cluster_dict[int(cluster)] = []
            cluster_dict[cluster].append(i)

        # Update self.data with cluster info
        for cluster, vectors in cluster_dict.items():
            for vector_index in vectors:
                self.data[vector_index]["cluster_id"] = cluster

        # Prepare self.cluster_data
        for cluster, vectors in cluster_dict.items():
            thumbnail_video_path = self.data[vectors[0]]["video_path"]
            thumbnail_time_start = self.data[vectors[0]]["time_start"]
            thumbnail = self.get_frame_at_time(thumbnail_video_path, thumbnail_time_start)

            self.cluster_data[cluster] = {
                "thumbnail": thumbnail,
                "frequency": len(vectors),
                "time_elapsed": len(vectors) * 2,
                "cluster_clip_id": vectors[0]
            }

    def get_frame_at_time(self, video_path, timestamp):
        video = cv2.VideoCapture(video_path)
        video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        success, frame = video.read()
        if not success:
            return None
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    
    def get_cluster_clip(self, cluster_id):
        cluster_clip_id = self.cluster_data[cluster_id]["cluster_clip_id"]
        return self.data[cluster_clip_id]

    def get_cluster(self, id):
        return self.cluster_data[id]

    def get_cluster_data(self):
        return self.cluster_data

    def get_data(self):
        return self.data

    def print_data(self):
        print(self.data)
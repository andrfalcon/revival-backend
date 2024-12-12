import os
from typing import Dict
from heapq import nlargest
from moviepy.editor import VideoFileClip

# TODO:
# Fix moviepy save video clip # moviepy 1.0.3 (DONE)
# Create yml file with conda
# Dockerize
# Share with marius

def save_video_clip(video_path, start_time, end_time, output_folder, clip_name):
    try:                    
        output_path = os.path.join(output_folder, clip_name)
        
        with VideoFileClip(video_path) as video:
            clip = video.subclip(float(start_time), float(end_time))
            clip.write_videofile(output_path, 
                               codec='libx264', 
                               audio_codec='aac',
                               temp_audiofile='temp-audio.m4a',
                               remove_temp=True)
            
        return output_path
        
    except Exception as e:
        raise Exception(f"Error saving video clip: {str(e)}")

def most_repeated_clusters(data: Dict[int, Dict[str, int]], n: int) -> Dict[int, Dict[str, int]]:
    # If n is greater than or equal to dictionary length, return original
    if n >= len(data):
        return data
    
    top_n_keys = nlargest(n, 
                         data.keys(),
                         key=lambda k: data[k]['frequency'])
    
    return {k: data[k] for k in top_n_keys}
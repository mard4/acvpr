# global imports
import cv2
import numpy as np
import os

# strong typing
from pathlib import Path
from typing import List
import glob


def main():
    """required variables are {pt_videos} and {pt_images}"""
    pt_videos = 'Videos'
    pt_images = 'frames_1fps'
    convert_videos_to_images(pt_videos=pt_videos, pt_images=pt_images)
    pass


def convert_videos_to_images(pt_videos: Path, pt_images: Path):
    """convert all videos from {pt_videos} to images saved to {pt_images}"""
    create_directory(pt=pt_images)
    ls_videos = glob.glob(os.path.join(pt_videos, '*.mp4'))
    ls_videos.sort()
    
    for str_video in ls_videos:
        pt_video = os.path.join(str_video)
        pt_image = os.path.join(pt_images, str_video.split("/")[-1].split('.')[0])
        num = int(pt_image.split('Hei-Chole')[1])
        pt_image = f'video_{num:02}'
        pt_image = os.path.join(pt_images, pt_image)

        create_directory(pt=pt_image)
        convert_video_to_image(pt_video=pt_video, pt_image=pt_image)


def convert_video_to_image(pt_video: Path, pt_image: Path):
    """convert a single video from {pt_video} to images saved to {pt_image}"""
    video_capture = cv2.VideoCapture(str(pt_video))
    int_frames_per_second: int = np.ceil(video_capture.get(cv2.CAP_PROP_FPS))  # ceiling function to ensure integer
    print('Extracting frames from ', pt_video)

    int_frame: int = 0
    while video_capture.isOpened():
        bool_success, np_frame_matrix = video_capture.read()
        if bool_success:
            if int_frame % int_frames_per_second == 0:
                pt_image_frame = os.path.join(pt_image, f"{int(int_frame / int_frames_per_second):05}.png")
                cv2.imwrite(str(pt_image_frame), np_frame_matrix)
        else:
            break
        int_frame += 1
    video_capture.release()
    print(f"{pt_video} successfully converted to {int_frame} images.")


def create_directory(pt: Path):
    """create a directory for a given {path} if it does not already exist"""
    if not os.path.exists(pt):
        os.mkdir(pt)


if __name__ == "__main__":
    main()
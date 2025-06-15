import skvideo.io
import numpy as np
import os, sys
import glob
from tqdm import tqdm
from pathlib import Path


def videos_to_imgs(output_path="/Videos/output",
                   input_path="/Videos/input",
                   pattern="*.mp4",
                   fps=25):
    output_path = Path(output_path)
    input_path = Path(input_path)

    dirs = list(input_path.glob(pattern))
    dirs.sort()
    output_path.mkdir(exist_ok=True)

    for i, vid_path in enumerate(tqdm(dirs)):
        file_name = vid_path.stem
        out_folder = output_path / file_name
        # or .avi, .mpeg, whatever.
        out_folder.mkdir(exist_ok=True)
        os.system(
            f'ffmpeg -i {vid_path} -vf "scale=250:250,fps=25" {out_folder/file_name}_%06d.png'
        )
        print("Done extracting: {}".format(i + 1))


# if __name__ == "__main__":
#     videos_to_imgs(output_path=None, input_path=None)

if __name__ == "__main__":
    # Path to your raw .mp4 files
    input_video_path = "Adv.CV/TeCNO_ver2/TeCNO/raw_cholec80_videos"

    # Path where the script will save the image folders (video01, video02, etc.)
    output_images_path = "Adv.CV/TeCNO_ver2/TeCNO/cholec80_processed/cholec_split_250px_25fps"

    videos_to_imgs(output_path=output_images_path, input_path=input_video_path)
import os
import os.path as osp
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm


SIZE = (920,540)
DATA_PATH = './MISAW/original-data' # root to the original data from https://www.synapse.org/Synapse:syn21776936/wiki/601700
SAVE_PATH = './MISAW/dataset/' # Path to save dataset frames 
ANN_PATH = './MISAW/annotations' # Path to save raw annotations
if not os.path.exists(ANN_PATH):
    os.makedirs(ANN_PATH)

splits = {'train': sorted(os.listdir(osp.join(DATA_PATH, 'train', 'Video'))),
          'test': sorted(os.listdir(osp.join(DATA_PATH, 'test', 'Video')))}
idx=0
for split in tqdm(splits.keys()):
    videos = splits[split]
    filenames = []
    phases = []
    for video in tqdm(videos):
        # read annotation file for the video
        idx += 1
        case_name = 'CASE{0:03d}'.format(int(idx))
        save_path = os.path.join(SAVE_PATH, case_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        video_name =  video.replace('.mp4', '_annotation.txt')
        df = pd.read_csv(osp.join(DATA_PATH, split, 'Procedural decription', video_name), sep='\t')
        # get the video information
        cv2video = cv2.VideoCapture(osp.join(DATA_PATH, split, 'Video', video))
        width  = cv2video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cv2video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        framecount = cv2video.get(cv2.CAP_PROP_FRAME_COUNT ) 
        fps = int(cv2video.get(cv2.CAP_PROP_FPS))
        assert int(framecount) == len(df)
        count = 0
        phases_anns = []
        steps_anns = []
        files=[]
        # 
        while count < framecount:
            hasFrames, cv2image = cv2video.read()
            cv2image = cv2.resize(cv2image, SIZE) #[:,:460]
            name = os.path.join(save_path, '{0:05d}.jpg'.format(count))
            cv2.imwrite(name, cv2image)
            files.append(name)
            phases_anns.append(df['Phase'].iloc[count])
            steps_anns.append(df['Step'].iloc[count])
            count += 1
        ann = {'filename': files,
               'phase': phases_anns,
               'step': steps_anns}
        ann = pd.DataFrame(ann)
        ann.to_csv(osp.join(ANN_PATH, case_name+'.csv'), index=False)
        
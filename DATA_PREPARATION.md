## Data Preparation

### Pre-Trained Models

You can download pre-trained models for all modules from this [Google Drive link](https://drive.google.com/drive/folders/1JMTtdFe8GbAeC6HiCwuvSAxNZpJsnqt-?usp=drive_link).  After downloading, place the files in a folder named `model_weights` within your project directory.

### Data Annotations and Frame Lists

Data annotations and frame lists are available [here](https://drive.google.com/drive/folders/10lhMDJeAdqZbtostIwph3Nd4KCltAFmr?usp=drive_link). Refer to the structure explained below for proper organization. Each dataset folder should include a `frames` directory containing the respective dataset's frames.

### Datasets

To download the datasets, use the following links:

- [MISAW dataset](https://www.synapse.org/Synapse:syn21776936/wiki/601700)  
- [GraSP dataset](https://github.com/BCV-Uniandes/GraSP)  
- [Cholec80 dataset](https://camma.unistra.fr/datasets/)  
- [HeiChole dataset](https://www.synapse.org/Synapse:syn25101790/wiki/610856)  

You may find these [formatting scripts](/utils) useful for sampling original videos into frames.


To run the models on a specific dataset, ensure your file structure follows this format:

```tree
MuST
|
|--configs
|   ...
|--data
|   |--dataset (GraSP, heichole, cholec80, misaw)
|       |--annotations (Subfolder structure varies by dataset. In this example, the dataset includes fold1, fold2, and training splits)
|       |   |--train
|       |   |   |--train.json
|       |   |   |--test.json
|       |   |--fold1
|       |   |   ...
|       |   |--fold2
|       |       ...
|       |--frame_lists
|       |   |--fold1.csv
|       |   |--fold2.csv
|       |   |--train.csv
|       |   |--test.csv
|       |
|       |--frames
|       |   |--video01
|       |   |   |--000000.jpg
|       |   |   |--000000.jpg
|       |   |   ...
|       |   |--video01
|       |   |   ...
|       |   ...
|--model_weights
|--run_files
|--must
|--tools
```

Feel free to use soft/hard linking to other paths or to modify the directory structure, the names, or the locations of the files, but then you may also have to modify the .yaml config files or the bash running scripts. 

## Annotation File Structure

MuST requires a COCO-format JSON annotation files containing the keys: ```phases_categories```, ```images```, and ```annotations```.

Each image dictionary in the ```images``` key, must include the image id, name of the frame, width, height the name of the video, and the number of the frame. For example, for the frame 00000.jpg from video01, the image dictionary entry should look like this:

```tree

{'id': 0, 
'file_name': 'video01/000000.jpg',
'width': 1920, 
'height': 1080, 
'date_captured': '', 
'license': 1, 'coco_url': '', 
'flickr_url': '', 
'video_name': 'video01', 
'frame_num': 0}
```
Each dictionary in the ```annotations``` key, must include the annotation id, image id, name of the frame, and the frame's category in the 'phases' key. For example, for the frame 00000.jpg from video01, the annotation dictionary entry should be:

```tree

{'id': 0, 
'image_id': 0, 
'image_name': 'video01/000000.jpg', 
'phases': 5}
```
## Frame List Structure

To build the frame windows in the dataloader, you need a CSV file listing frames in ascending order. This file should contain 4 columns: the video name, a numerical video identifier, the frame number within the video, and the frame name (e.g., 00000.jpg). For example, in a dataset with 2 videos, the CSV file would look like this: 

```tree

video01 1 0 video01/00000.jpg
video01 1 1 video01/00001.jpg
video01 1 2 video01/00002.jpg
video01 1 3 video01/00003.jpg
video01 1 4 video01/00004.jpg
            .
            .
            .
video02 2 0 video02/00000.jpg
video02 2 1 video02/00001.jpg
video02 2 2 video02/00002.jpg
video02 2 3 video02/00003.jpg
video02 2 4 video02/00004.jpg
            .
            .
            .
video02 2 9996 video02/09996.jpg
video02 2 9997 video02/09997.jpg
video02 2 9998 video02/09998.jpg
video02 2 9999 video02/09999.jpg
video02 2 10000 video02/10000.jpg
```

This structure ensures the frames are properly ordered and compatible with the dataloader.

## Custom Dataset

If you want to run the model on a custom dataset, you can refer to the dataset template provided at [must/datasets/custom_dataset.py](must/datasets/). 

To evaluate your metrics in your custom dataset, follow the instructions in [must/utils/meters.py](must/utils/).

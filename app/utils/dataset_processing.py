# app/utils/dataset_processing.py
import os
import glob

def prepare_classification_dataset(source_dir, target_dir):
    """
    Prepares the video classification dataset by creating lists of video paths
    organized by class and split (train/test).

    Args:
        source_dir (str): Path to the directory containing video data.
                            Expected structure: source_dir/Train/Class/videos,
                                              source_dir/Test/Class/videos
        target_dir (str):  (Not used to save data.  It's kept for backward compatibility.

    Returns:
        dict: A dictionary containing lists of video file paths,
              organized by split (train/test) and class.
              Example:
              {
                  'train': {
                      'Normal': ['path/to/video1.avi', 'path/to/video2.avi', ...],
                      'Violence': [...],
                      'Weaponized': [...]
                  },
                  'test': {
                      'Normal': [...],
                      'Violence': [...],
                      'Weaponized': [...]
                  }
              }
    """
    dataset = {}
    for split in ['Train', 'Test']:
        dataset[split.lower()] = {}
        split_dir = os.path.join(source_dir, split)
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir): #only process the folders
                video_files = glob.glob(os.path.join(class_dir, '*.*v*'))  # This will capture .avi, .mp4, .mov, etc. #adapt based on your video extensions
                dataset[split.lower()][class_name] = video_files
    return dataset
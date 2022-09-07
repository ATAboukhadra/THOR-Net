import torch
import os

from itertools import product
from typing import List, Optional
from .h2o_preprocessing_utils import MyPreprocessor

def get_tar_lists(base_path: str, data_components: List[str], subjects: Optional[List[int]] = None,
                  scenes: Optional[List[str]] = None, sequences: Optional[List[int]] = None,
                  cameras: Optional[List[int]] = None):
    """ Create for each requested component of the dataset a sequence of tar file shards that include the specified
    subjects, scenes, objects and cameras.

    :param base_path: The base path where the shards are stored.
    :param data_components: A list of the components of the datasets that shall be used. The valid options are:
       'rgb', 'rgb256', 'depth', 'annotations'.
    :param subjects: A list of the subjects that shall be included. The valid options are: 1, 2, 3, 4.
    :param scenes: A list of the scenes that shall be included. The options are: 'h1', 'h2', 'k1', 'k2', 'o1', 'o2'.
    :param sequences: A list of the sequences that shall be included. The options are: 0, 1, 2, 3, 4, 5, 6, 7.
    :param cameras: The cameras that shall be included. The options are: 0, 1, 2, 3, 4.
    :return: For each requested component of the dataset a list of tar files. (Returned as a list of lists.)
    """
    if subjects is None:
        subjects = [1, 2, 3, 4]
    if scenes is None:
        scenes = ['h1', 'h2', 'k1', 'k2', 'o1', 'o2']
    if sequences is None:
        sequences = [0, 1, 2, 3, 4, 5, 6, 7]
    if cameras is None:
        cameras = [0, 1, 2, 3, 4]
    tar_file_lists = []
    # Create tar list. Skip combinations that do not exist.
    for data_component in data_components:
        tar_file_list = [os.path.join(base_path, f'subject{subject}_{scene}_{obj}_cam{camera}_{data_component}.tar')
                         for subject, scene, obj, camera
                         in product(subjects, scenes, sequences, cameras)
                         if (data_component != 'rgb256' or (data_component == 'rgb256' and camera == 4)) and
                         not (obj == 7 and ((subject == 1 and scene == "k2") or
                                            (subject == 2 and scene in ["h2", "k1", "k2", "o1", "o2"]) or
                                            (subject == 4 and scene in ["k1", "o2"])))]
        tar_file_lists.append(sorted(tar_file_list))
    
    # for k in tar_file_lists[0]:
    #     print(k)
    return tar_file_lists


def load_tar_split(data_dir, split):

    if split == 'train':
    
        input_tar_lists = get_tar_lists(data_dir, ['rgb'], subjects=[1, 2], cameras=[4])
        additional_input_tar_lists = get_tar_lists(data_dir, ['rgb'], subjects=[3], scenes=['h1', 'h2', 'k1'], cameras=[4])
        for i in range(len(input_tar_lists)):
            input_tar_lists[i].extend(additional_input_tar_lists[i])

        annotation_tar_files = get_tar_lists(data_dir, ['annotations'], subjects=[1, 2], cameras=[4])[0]
        annotation_tar_files.extend(get_tar_lists(data_dir, ['annotations'], subjects=[3], scenes=['h1', 'h2', 'k1'], cameras=[4])[0])

    elif split == 'val':
        input_tar_lists = get_tar_lists(data_dir, ['rgb'], subjects=[3], scenes=['k2', 'o1', 'o2'], cameras=[4])
        annotation_tar_files = get_tar_lists(data_dir, ['annotations'], subjects=[3], scenes=['k2', 'o1', 'o2'], cameras=[4])[0]

    else:
        input_tar_lists = get_tar_lists(data_dir, ['rgb'], subjects=[4], cameras=[4])
        annotation_tar_files = get_tar_lists(data_dir, ['annotations'], subjects=[4], cameras=[4])[0]

    return input_tar_lists, annotation_tar_files
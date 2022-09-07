import io
import os
from typing import Any, Callable, Dict

import numpy as np
import torch
from torch.utils.data.datapipes.datapipe import DataChunk
from torch.utils.data.datapipes.utils.common import StreamWrapper
from torch.utils.data.datapipes.utils.decoder import ImageHandler


def group_fn(file_path: str):
    """ Group files into samples based on the file name up to the first '.'.

    This will work with any dataset that follows the file naming convention of WebDataset:
    sample.sample_component.file_extension

    :param file_path: The path to the file for which to extract the sample name/ID.
    :return: The sample name/ID used to group the files into samples.
    """
    bn = os.path.basename(file_path[0])
    return bn.split(".")[0]


def collate_batch_as_tuple(samples):
    """ Creates batches from individual samples.

    Adds a batch dimension to tensors and stacks the individual tensors in that dimension.
    Other data types are simply placed in lists. All aggregated dataset elements are then placed in a tuple.

    :param samples: The samples that shall be combined into a batch.
    :return: The batched data in two tuples: The first tuple contains the filenames for each component.
     The second tuple contains the actual data.
    """
    sample_size = len(samples[0])
    file_name_list = []
    data_object_list = []
    file_name_list = [sample[0] for sample in samples]
    for element_idx in range(1, sample_size):
        # Split the tuple into a list of the filenames and a list of the actual data elements
        data_objects = [sample[element_idx] for sample in samples]
        # stack torch tensors
        if isinstance(data_objects[0], torch.Tensor):
            data_objects = torch.stack(data_objects)
        # stack numpy arrays
        elif isinstance(data_objects[0], np.ndarray):
            data_objects = np.stack(data_objects)
        data_object_list.append(data_objects)
    return tuple(file_name_list), tuple(data_object_list)

def collate_batch_as_dict(samples):
    output_list = []
    for sample in samples:
        sample_dict = {
            'path': sample[0],
            'inputs': sample[1],
            'keypoints2d': sample[2],
            'keypoints3d': sample[3].unsqueeze(0),
            'mesh2d': sample[4],
            'mesh3d': sample[5].unsqueeze(0),
            'boxes': sample[6],
            'labels': sample[7],
            'keypoints': sample[8]
        }
        output_list.append(sample_dict)
    return output_list



def decoder_key_fn(file_path: str) -> str:
    """
    Extract everything after the last but one dot from the file name (path) as a string to be used as the key for
    selecting the proper decoder for the file.

    :param file_path: A full path or filename.
    :return: The key to be used for selecting the proper decoder for the file.
    """
    bn = os.path.basename(file_path)
    key = str.join(".", bn.split(".")[-2:])
    return key


class GrayscaleDecoder:
    def __init__(self):
        # torchl is broken (due to missing channel dimension), use numpy format instead
        self.grayscale_decoder = ImageHandler('l')

    def __call__(self, extension, data) -> torch.Tensor:
        # decode to numpy
        np_img = self.grayscale_decoder(extension, data)
        # add missing channel dimension if case of single channel (grayscale) images
        np_img = np_img[np.newaxis, :, :]
        # manually convert to torch tensor format
        return torch.tensor(np_img, dtype=torch.float32)



class Decoder:
    def __init__(self, decoders: Dict[str, Callable[[str, bytes], Any]]):
        self.decoders = decoders

    @staticmethod
    def _is_stream_handle(data):
        obj_to_check = data.file_obj if isinstance(data, StreamWrapper) else data
        return isinstance(obj_to_check, io.BufferedIOBase) or isinstance(obj_to_check, io.RawIOBase)

    def __call__(self, sample: DataChunk):
        new_sample = []
        for file_path, data in sample:
            key = decoder_key_fn(file_path)
            component, extension = os.path.splitext(key)
            # remove "."
            extension = extension[1:]

            if Decoder._is_stream_handle(data):
                ds = data
                # The behavior of .read can differ between streams (e.g. HTTPResponse), hence this is used instead
                data = b"".join(data)
                ds.close()

            if component in self.decoders.keys():
                new_sample.append((file_path, self.decoders[component](extension, data)))
            else:
                # Pass data on without decoding
                new_sample.append((file_path, data))
        return DataChunk(tuple(new_sample))

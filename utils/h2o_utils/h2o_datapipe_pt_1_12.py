import os
import random
from typing import List, Tuple

import torchdata.datapipes as dp
from torch.utils.data.datapipes.datapipe import DataChunk
from torch.utils.data.datapipes.utils.decoder import ImageHandler

from .h2o_datapipe_helpers import Decoder, GrayscaleDecoder, group_fn
from .h2o_annotation_decoding import *


def merge_fn(sample: Tuple) -> DataChunk:
    """
    Merge a varying number of inputs (RGB, RGB256 and/or depth) with the annotations.
    :param sample: A tuple of the individual elements (from separate data pipes) that shall be merged.
    :return: The merged sample as a DataChunk (essentially a tuple).
    """
    # The data is stored in two nested tuples. The outer tuple contains the image data and the tuple containing the
    # annotations. We dissolve the inner tuple and integrate the elements into the outer tuple.
    merged_sample = tuple(zip(*sample[:-1], *sample[-1]))
    # Convert to DataChunks, because this is what outer pipeline operations expect
    merged_sample = DataChunk(zip(*merged_sample))
    return merged_sample


class Filter:
    def __init__(self, components: List[str]):
        self.keep_list = components

    def __call__(self, data):
        file_path = data[0]
        key = os.path.basename(file_path).split(".")[-2]
        return key in self.keep_list


def make_decoder():
    decoder_map = {'cam_pose': decode_cam_pose,
                   'hand_pose': decode_hand_pose,
                   'hand_pose_mano': decode_hand_pose_mano,
                   'obj_pose': decode_obj_pose,
                   'obj_pose_rt': decode_obj_pose_rt,
                   'action_label': decode_label,
                   'verb_label': decode_label,
                   'rgb': ImageHandler('torchrgb'),
                   'depth': GrayscaleDecoder()}
    return Decoder(decoder_map)


def create_datapipe(input_tars_list: List[List[str]],
                    annotation_tars: List[str],
                    annotation_components: List[str],
                    shuffle_buffer_size: int,
                    shuffle_shards: bool = True) -> dp.iter.IterDataPipe:
    """ Create an iterable data pipe for the H2O dataset.

    :param input_tars_list: For each type of input a list of tar files of the shards of the dataset for that component.
    :param annotation_tars: A list of tar files with annotations.
    :param annotation_components: A list of the annotations that shall be included. Annotations not in the list will be
     discarded.
    :param shuffle_buffer_size: The size of the shuffle buffer (for each worker). If possible a size should be chosen
     that is larger than the number of samples in a single shard.
     Ideally: shuffler_buffer_size = samples_per_shard * batch_size
    :param shuffle_shards: Whether to shuffle the shards for improved randomness. Only needed for training.
    :return: The dataset as an IterDataPipe.
    """
    num_components = len(annotation_components)
    min_num_components = len([x for x in annotation_components if x not in ["action_label", "verb_label"]])
    input_data_pipe_count = len(input_tars_list)
    pipeline_count = input_data_pipe_count + 1
    # zip tar file sequences. they need to be shuffled and sharded together.
    zipped_shard_list = list(zip(*input_tars_list, annotation_tars))
    if shuffle_shards:
        # Shuffle the order of the tar files. This is necessary to get good randomness during training.
        # Warning: A new random shuffle is needed in every epoch. Otherwise, the order of samples is too deterministic.
        random.shuffle(zipped_shard_list)
    # Wrap the zipped shard sequence to create an iterable pipeline
    pipe = dp.iter.IterableWrapper(zipped_shard_list)
    # apply sharding to distribute work to workers
    pipe = pipe.sharding_filter()
    # unzip the tar sequence to create separate pipelines for the individual tar sequences
    pipes = pipe.unzip(sequence_length=pipeline_count, buffer_size=pipeline_count)
    # open and read the tar files in each sequence separately
    pipes = [dp.iter.FileOpener(pipe, mode="b").load_from_tar() for pipe in pipes]
    # Filter annotations
    pipes[-1] = pipes[-1].filter(Filter(annotation_components))
    # Group annotations together. Depending on the camera there are at least 5 and up to 7 annotations per sample.
    pipes[-1] = pipes[-1].groupby(group_key_fn=group_fn, buffer_size=num_components, group_size=num_components,
                                  guaranteed_group_size=min_num_components)
    # merge the pipelines back together
    pipe = pipes[0].zip(*pipes[1:])
    # fuse the individual tuples
    pipe = pipe.map(merge_fn)
    # Shuffle the files in memory. Requires memory according to: num_workers * shuffle_buffer_size * size_of_a_sample.
    # Since this is done before decoding, the memory consumption is determined by file size not the decoded size.
    pipe = pipe.shuffle(buffer_size=shuffle_buffer_size)
    # Decode the individual files of each sample. Images are converted to torch tensors with values between 0.0 and 1.0.
    pipe = pipe.map(make_decoder())
    return pipe

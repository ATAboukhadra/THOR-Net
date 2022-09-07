from typing import List, Union
import torch

def decode_text_to_floats(extension: str, data: bytes) -> Union[torch.Tensor, List[torch.Tensor]]:
    """ Decode text files containing floating point numbers separated by whitespaces.

    :param data: A stream of bytes representing the content of the file.
    :return: The decoded values as a list of torch tensors or a single tensor if there is only one line.
    """
    # Decode byte stream to UTF-8 text
    lines = [line.split(' ') for line in data.decode("utf-8").splitlines()]
    # Remove empty strings caused by additional whitespaces in input
    for i in range(len(lines)):
        lines[i] = [x for x in lines[i] if x != ""]
    # Convert to float tensors
    tensors = []
    for i in range(len(lines)):
        tensors.append(torch.tensor([float(x) for x in lines[i]]))
    if len(tensors) == 1:
        tensors = tensors[0]
    return tensors


def decode_label(extension: str, data: bytes) -> torch.Tensor:
    class_str = data.decode("utf-8")
    class_id = torch.tensor(int(class_str))
    return class_id


def decode_cam_pose(extension: str, data: bytes):
    cam_matrix = decode_text_to_floats(extension, data).reshape(4, 4)
    return cam_matrix


def decode_hand_pose(extension: str, data: bytes) -> torch.Tensor:
    """ Decode hand pose data to a 42x3 tensor """
    data = decode_text_to_floats(extension, data)

    # Initialize placeholders for left & right hand poses
    left_hand = torch.zeros((21, 3))
    right_hand = torch.zeros((21, 3))

    # Values at 0th and 64th position are flags that represent the presence of a left or a right hand in the sample,
    # respectively. The remaining values represent left- & right-hand poses, respectively.
    # Each pose has 21 keypoints in the format x,y,z.
    if data[0] == 1:
        left_hand = torch.reshape(data[1:64], (21, 3))
    if data[64] == 1:
        right_hand = torch.reshape(data[65:129], (21, 3))

    # Scaling factor to transform 3D points from meters to millimeters
    scaling_factor = 1000

    # Concatenate both poses to return a 42x3 tensor
    hands = torch.cat([left_hand, right_hand]) * scaling_factor

    return hands


def decode_hand_pose_mano(extension: str, data: bytes):
    # The data contains 62 values for the left followed by 62 values for the right hand.
    # The meaning of the values for each hand are:
    # 1 (whether annotated or not, 0: not annotate 1: annotate)
    # 3 translation values
    # 48 pose values
    # 10 shape values
    values = decode_text_to_floats(extension, data)
    return values


def decode_obj_pose(extension: str, data: bytes) -> torch.Tensor:
    """ Reshapes object pose to 8x3 and append a row that holds the object label """

    data = decode_text_to_floats(extension, data)

    # Scaling factor to transform 3D points from meters to millimeters
    scaling_factor = 1000

    # Initialize a placeholder for object pose
    obj_pose = torch.zeros((8, 3))

    # First element specifies the label of the object (0 for no object)
    # Only the 8 corners of the object are being decoded. They lie in the range of 4 to 28.
    # The other values such as the center and mid edge points are currently not being decoded.
    if data[0] > 0:
        obj_pose = torch.reshape(data[4:28], (8, 3)) * scaling_factor

    # Add a 9th row that represents the class label
    obj_label = torch.tensor([[data[0], 0, 0]])
    obj_pose = torch.cat([obj_pose, obj_label])

    return obj_pose


def decode_obj_pose_rt(extension: str, data: bytes):
    """ Decode the object pose rt data """
    obj_cam_mat = decode_text_to_floats("", data)
    # Drop object id and reshape remaining data to a 4x4 camera matrix
    obj_cam_mat = obj_cam_mat[1:].reshape(4, 4)
    return obj_cam_mat


def decode_cam_intrinsics(data: bytes):
    """ Reshapes camera intrinsics to a 3x3 matrix """

    cam_intr = decode_text_to_floats("", data)
    cam_intr = torch.tensor([
        [cam_intr[0], 0, cam_intr[2]],
        [0, cam_intr[1], cam_intr[3]],
        [0, 0, 1]])

    return cam_intr

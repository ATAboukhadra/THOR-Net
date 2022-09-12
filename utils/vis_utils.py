from __future__ import print_function, unicode_literals
import numpy as np
import json
import os
import time
import skimage.io as io
import pickle
import math
import sys
import matplotlib.pyplot as plt
import cv2
import pymeshlab
from manopth.manolayer import ManoLayer


""" General util functions. """
def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg


def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def showHandJoints(imgInOrg, gtIn, filename=None, dataset_name='ho', mode='pred'):
    '''
    Utility function for displaying hand annotations
    :param imgIn: image on which annotation is shown
    :param gtIn: ground truth annotation
    :param filename: dump image name
    :return:
    '''
    import cv2

    imgIn = np.copy(imgInOrg)
    # Set color for each finger

    joint_color_code = [[139, 53, 255],
                        [0, 56, 255],
                        [43, 140, 237],
                        [37, 168, 36],
                        [147, 147, 0],
                        [70, 17, 145]]

    if mode == 'gt':
        joint_color_code = [[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]]

    limbs = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [0, 5],
             [5, 6],
             [6, 7],
             [7, 8],
             [0, 9],
             [9, 10],
             [10, 11],
             [11, 12],
             [0, 13],
             [13, 14],
             [14, 15],
             [15, 16],
             [0, 17],
             [17, 18],
             [18, 19],
             [19, 20]
             ]

    cf = 35 

    gtIn = np.round(gtIn).astype(np.int)

    if gtIn.shape[0]==1:
        imgIn = cv2.circle(imgIn, center=(gtIn[0][0], gtIn[0][1]), radius=3, color=joint_color_code[0],
                             thickness=-1)
    else:
        max_length=500
        for joint_num in range(gtIn.shape[0]):

            color_code_num = (joint_num // 4)
            joint_color = list(map(lambda x: x + cf * (joint_num % 4), joint_color_code[color_code_num]))[::-1]    
            
            cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color, thickness=-1)
        
        for limb_num in range(len(limbs)):
            x1 = gtIn[limbs[limb_num][0], 1]
            y1 = gtIn[limbs[limb_num][0], 0]
            x2 = gtIn[limbs[limb_num][1], 1]
            y2 = gtIn[limbs[limb_num][1], 0]
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if length < max_length and length > 5:
                deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
                polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                           (int(length / 2), 2),
                                           int(deg),
                                           0, 360, 1)
                color_code_num = limb_num // 4

                limb_color = list(map(lambda x: x  + cf * (limb_num % 4), joint_color_code[color_code_num]))[::-1]


                cv2.fillConvexPoly(imgIn, polygon, color=limb_color)

    if filename is not None:
        cv2.imwrite(filename, cv2.cvtColor(imgIn, cv2.COLOR_RGB2BGR))

    return imgIn

def showObjJoints(imgIn, gtIn, estIn=None, filename=None, upscale=1, lineThickness=2):
    '''
    Utility function for displaying object annotations
    :param imgIn: image on which annotation is shown
    :param gtIn: ground truth annotation
    :param estIn: estimated keypoints
    :param filename: dump image name
    :param upscale: scale factor
    :param lineThickness:
    :return:
    '''
    import cv2
    jointConns = [[0, 1, 3, 2, 0], [4, 5, 7, 6, 4], [0, 4], [1, 5], [2, 6], [3,7]]
    jointColsGt = (255,255,0)
    jointColsEst  = (0, 0, 0)

    # draws lines connected using jointConns
    img = np.zeros((imgIn.shape[0], imgIn.shape[1], imgIn.shape[2]), dtype=np.uint8)
    img[:, :, :] = (imgIn).astype(np.uint8)

    img = cv2.resize(img, (upscale * imgIn.shape[1], upscale * imgIn.shape[0]), interpolation=cv2.INTER_CUBIC)
    if gtIn is not None:
        gt = gtIn.copy() * upscale
    if estIn is not None:
        est = estIn.copy() * upscale

    for i in range(len(jointConns)):
        for j in range(len(jointConns[i]) - 1):
            jntC = jointConns[i][j]
            jntN = jointConns[i][j+1]
            if gtIn is not None:
                cv2.line(img, (int(gt[jntC,0]), int(gt[jntC,1])), (int(gt[jntN,0]), int(gt[jntN,1])), jointColsGt, lineThickness)
            if estIn is not None:
                cv2.line(img, (int(est[jntC,0]), int(est[jntC,1])), (int(est[jntN,0]), int(est[jntN,1])), jointColsEst, lineThickness)

    if filename is not None:
        cv2.imwrite(filename, img)

    return img

def draw_bb(img, bb, color):
    """ Show bounding box on the image"""
    bb_img = np.copy(img)

    # print(bb, bb_img.shape, bb_img.dtype)
    bb = bb.astype(int)
    cv2.rectangle(bb_img, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
    return bb_img

def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)

def plot3dVisualize(ax, m, faces, flip_x=False, c="b", alpha=0.1, camPose=np.eye(4, dtype=np.float32), isOpenGLCoords=False):
    '''
    Create 3D visualization
    :param ax: matplotlib axis
    :param m: mesh
    :param flip_x: flix x axis?
    :param c: mesh color
    :param alpha: transperency
    :param camPose: camera pose
    :param isOpenGLCoords: is mesh in openGL coordinate system?
    :return:
    '''
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if hasattr(m, 'r'):
        verts = np.copy(m.r) * 1000
    elif hasattr(m, 'v'):
        verts = np.copy(m.v) * 1000
    elif isinstance(m, np.ndarray): # In case of an output of a Mano layer (no need to scale)
        verts = np.copy(m)
    else:
        raise Exception('Unknown Mesh format')
    vertsHomo = np.concatenate([verts, np.ones((verts.shape[0],1), dtype=np.float32)], axis=1)
    verts = vertsHomo.dot(camPose.T)[:,:3]

    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if isOpenGLCoords:
        verts = verts.dot(coordChangeMat.T)

    ax.view_init(elev=120, azim=-90)
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    if c == "b":
        face_color = (141 / 255, 184 / 255, 226 / 255)
        face_color = np.tile(np.array([[0., 0., 1., 1.]]), [verts.shape[0], 1])
        edge_color = (0 / 255, 0 / 255, 112 / 255)
    elif c == "r":
        face_color = (226 / 255, 141 / 255, 141 / 255)
        face_color = np.tile(np.array([[1., 0., 0., 1.]]), [verts.shape[0], 1])
        edge_color = (112 / 255, 0 / 255, 0 / 255)
    elif c == "viridis":
        face_color = plt.cm.viridis(np.linspace(0, 1, faces.shape[0]))
        edge_color = None
        edge_color = (0 / 255, 0 / 255, 112 / 255)
    elif c == "plasma":
        face_color = plt.cm.plasma(np.linspace(0, 1, faces.shape[0]))
        edge_color = None
        # edge_color = (0 / 255, 0 / 255, 112 / 255)
    else:
        face_color = c
        edge_color = c

    mesh.set_edgecolor(edge_color)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    cam_equal_aspect_3d(ax, verts, flip_x=flip_x)
    # plt.tight_layout()

def show3DHandJoints(ax, verts, mode='pred', isOpenGLCoords=False):
    '''
    Utility function for displaying hand 3D annotations
    :param ax: matplotlib axis
    :param verts: ground truth annotation
    '''

    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if isOpenGLCoords:
        verts = verts.dot(coordChangeMat.T)

    limbs = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [0, 5],
             [5, 6],
             [6, 7],
             [7, 8],
             [0, 9],
             [9, 10],
             [10, 11],
             [11, 12],
             [0, 13],
             [13, 14],
             [14, 15],
             [15, 16],
             [0, 17],
             [17, 18],
             [18, 19],
             [19, 20]
             ]


    joint_color_code = ['b', 'g', 'r', 'c', 'm']

    if mode == 'gt':
        joint_color_code = ['k'] * 5

    ax.view_init(elev=120, azim=-90)
    for limb_num in range(len(limbs)):
        x1 = verts[limbs[limb_num][0], 0]
        y1 = verts[limbs[limb_num][0], 1]
        z1 = verts[limbs[limb_num][0], 2]
        x2 = verts[limbs[limb_num][1], 0]
        y2 = verts[limbs[limb_num][1], 1]
        z2 = verts[limbs[limb_num][1], 2]
        ax.plot([x1, x2], [y1, y2], [z1, z2], color=joint_color_code[limb_num//4])

    x = verts[:,0]
    y = verts[:,1]
    z = verts[:,2]
    ax.scatter(x, y, z)

def show3DObjCorners(ax, verts, mode='pred', isOpenGLCoords=False):
    '''
    Utility function for displaying Object 3D annotations
    :param ax: matplotlib axis
    :param verts: ground truth annotation
    '''

    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if isOpenGLCoords:
        verts = verts.dot(coordChangeMat.T)

    jointConns = [[0, 1, 3, 2, 0], [4, 5, 7, 6, 4], [0, 4], [1, 5], [2, 6], [3,7]]
    
    for i in range(len(jointConns)):
        for j in range(len(jointConns[i]) - 1):
            jntC = jointConns[i][j]
            jntN = jointConns[i][j+1]
            if mode == 'gt':
                ax.plot([verts[jntC][0], verts[jntN][0]], [verts[jntC][1], verts[jntN][1]], [verts[jntC][2], verts[jntN][2]], color='k')
            else:    
                ax.plot([verts[jntC][0], verts[jntN][0]], [verts[jntC][1], verts[jntN][1]], [verts[jntC][2], verts[jntN][2]], color='y')

    x = verts[:,0]
    y = verts[:,1]
    z = verts[:,2]

    ax.scatter(x, y, z)

def show2DMesh(fig, ax, img, mesh2DPoints, gt=False, filename=None):
    ax.imshow(img)
    if gt:
        ax.scatter(mesh2DPoints[:, 0], mesh2DPoints[:, 1], alpha=0.3, s=20, color="black", marker='.')
    else:
        ax.scatter(mesh2DPoints[:778, 0], mesh2DPoints[:778, 1], alpha=0.3, s=20, marker='.')
        if mesh2DPoints.shape[0] > 778:
            ax.scatter(mesh2DPoints[778:, 0], mesh2DPoints[778:, 1], alpha=0.3, s=20, color="red", marker='.')
    
    # Save just the portion _inside_ the second axis's boundaries
    if filename is not None:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'{filename}', bbox_inches=extent)

def draw_confidence(image, keypoints, scores):
    keypoints = np.round(keypoints).astype(np.int)

    high_confidence = np.where(scores >= 2)[0]
    low_confidence = np.where(scores < 2)[0]
    
    for idx in high_confidence:
        cv2.circle(image, center=(keypoints[idx][0], keypoints[idx][1]), radius=3, color=[43, 140, 237], thickness=-1)
    for idx in low_confidence:
        cv2.circle(image, center=(keypoints[idx][0], keypoints[idx][1]), radius=3, color=[0, 0, 0], thickness=-1)
    
    return image

def plot_bb_ax(img, outputs, fig_config, subplot_id, plot_txt):
    fig, H, W = fig_config
    bb_image = np.copy(img)
    ax = fig.add_subplot(H, W, subplot_id)
    
    labels = list(outputs['labels'])
    
    if max(labels) > 1:
        required_bbs = [
            outputs['boxes'][labels.index(1)],
            outputs['boxes'][labels.index(2)],
            outputs['boxes'][labels.index(3)]
        ]
    else:
        required_bbs = outputs['boxes']
    
    for bb in required_bbs:
        bb_image = draw_bb(bb_image, bb, [229, 255, 204])    
    
    ax.title.set_text(plot_txt)
    ax.imshow(bb_image)

def plot_pose2d(img, outputs, idx, fig_config, subplot_id, plot_txt, center=None, dataset_name='h2o'):

    if dataset_name == 'h2o':
        cam_mat = np.array([[636.6593,   0.0000, 635.2839],
        [  0.0000, 636.2520, 366.8740],
        [  0.0000,   0.0000,   1.0000]]
        )
    else:
        cam_mat = np.array(
            [[617.343,0,      312.42],
            [0,       617.343,241.42],
            [0,       0,       1]
        ])

    keypoints3d = outputs['keypoints3d'][idx]
    
    if center is not None:
        keypoints = project_3D_points(cam_mat, keypoints3d + center, is_OpenGL_coords=False)
    else:
        keypoints = project_3D_points(cam_mat, keypoints3d, is_OpenGL_coords=False)

    fig, H, W = fig_config
    plt_image = np.copy(img)
    
    ax = fig.add_subplot(H, W, subplot_id)
    plt_image = showHandJoints(plt_image, keypoints[:21])
    
    # If pose is only 1 hand and object (HO3D)
    if keypoints.shape[0] == 29:
        plt_image = showObjJoints(plt_image, keypoints[21:])
    
    # If pose is only 2 hands and object (H2O)
    if keypoints.shape[0] == 50:
        plt_image = showHandJoints(plt_image, keypoints[21:42])
        plt_image = showObjJoints(plt_image, keypoints[42:50])
 
    ax.title.set_text(plot_txt)
    ax.imshow(plt_image)
    

def plot_pose3d(labels, fig_config, subplot_id, plot_txt, mode='pred', center=None, idx=0):
    
    fig, H, W = fig_config
    keypoints3d = labels['keypoints3d'][idx]
    if center is not None:
        keypoints3d += center

    ax = fig.add_subplot(H, W, subplot_id, projection="3d")
    
    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    show3DHandJoints(ax, keypoints3d[:21], mode=mode, isOpenGLCoords=True)
    
    # If pose is only 1 hand and object (HO3D)
    if keypoints3d.shape[0] == 29:
        show3DObjCorners(ax, keypoints3d[21:], mode=mode, isOpenGLCoords=True)
    
    # If pose is only 2 hands and object (H2O)
    if keypoints3d.shape[0] == 50:
        show3DHandJoints(ax, keypoints3d[21:42], mode=mode, isOpenGLCoords=True)
        show3DObjCorners(ax, keypoints3d[42:], mode=mode, isOpenGLCoords=True)

    ax.title.set_text(plot_txt)

def plot_mesh3d(outputs, right_hand_faces, obj_faces, fig_config, subplot_id, plot_txt, left_hand_faces=None, center=None, idx=0):
    
    fig, H, W = fig_config
    ax = fig.add_subplot(H, W, subplot_id, projection="3d")
    
    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    mesh3d = outputs['mesh3d'][idx][:, :3]
    num_verts = mesh3d.shape[0]

    if center is not None:
        mesh3d += center
    
    if num_verts <= 1778 :
        plot3dVisualize(ax, mesh3d[:778], right_hand_faces, flip_x=False, isOpenGLCoords=False, c="r")
    
    if num_verts == 1778:
        plot3dVisualize(ax, mesh3d[778:], obj_faces, flip_x=False, isOpenGLCoords=False, c="b")

    if num_verts > 1778:
        plot3dVisualize(ax, mesh3d[:778], left_hand_faces, flip_x=False, isOpenGLCoords=False, c="r")
        plot3dVisualize(ax, mesh3d[778:778*2], right_hand_faces, flip_x=False, isOpenGLCoords=False, c="g")
        plot3dVisualize(ax, mesh3d[778*2:], obj_faces, flip_x=False, isOpenGLCoords=False, c="b")

    cam_equal_aspect_3d(ax, mesh3d, flip_x=False)
    ax.title.set_text(plot_txt)

def plot_pose_heatmap(img, predictions, idx, center, fig_config, plot_id):

    # porject to 2D 
    cam_mat = np.array(
        [[617.343,0,      312.42],
        [0,       617.343,241.42],
        [0,       0,       1]
    ])
    
    keypoints3d = predictions['keypoints3d'][idx]
    keypoints = project_3D_points(cam_mat, keypoints3d + center, is_OpenGL_coords=False)
    keypoints = np.round(keypoints).astype(np.int)
    
    fig, H, W = fig_config
    ax = fig.add_subplot(H, W, plot_id)

    heatmap = create_heatmap(img, keypoints)

    ax.imshow(heatmap, cmap='viridis')

def create_heatmap(img, keypoints):

    heatmap = np.zeros_like(img[:, :, 0]) 
    r = 6
    translations = []
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            distance = i**2 + j**2
            if (distance <= r*r):
                translations.append((i, j, 1 - math.sqrt(distance)/r))
    for t in translations:
        if np.all(keypoints[:, 1] + t[0] < heatmap.shape[0]) and np.all(keypoints[:, 0] + t[1] < heatmap.shape[1]):
            heatmap[keypoints[:, 1] + t[0], keypoints[:, 0] + t[1]] = t[2] * 10
    
    return heatmap

class Minimal(object):
    def __init__(self, **kwargs):
        self.__dict__ = kwargs
        
def read_obj(filename):
    """ Reads the Obj file. Function reused from Matthew Loper's OpenDR package"""

    lines = open(filename).read().split('\n')

    d = {'v': [], 'f': []}

    for line in lines:
        line = line.split()
        if len(line) < 2:
            continue

        key = line[0]
        values = line[1:]

        if key == 'v':
            d['v'].append([np.array([float(v) for v in values[:3]])])
        elif key == 'f':
            spl = [l.split('/') for l in values]
            d['f'].append([np.array([int(l[0])-1 for l in spl[:3]], dtype=np.uint32)])

    for k, v in d.items():
        if k in ['v','f']:
            if v:
                d[k] = np.vstack(v)
            else:
                print(k)
                del d[k]
        else:
            d[k] = v

    result = Minimal(**d)

    return result


def db_size(set_name, version='v2'):
    """ Hardcoded size of the datasets. """
    if set_name == 'train':
        if version == 'v2':
            return 66034  # number of unique samples (they exists in multiple 'versions')
        elif version == 'v3':
            return 78297
        else:
            raise NotImplementedError
    elif set_name == 'evaluation':
        if version == 'v2':
            return 11524
        elif version == 'v3':
            return 20137
        else:
            raise NotImplementedError
    else:
        assert 0, 'Invalid choice.'

def load_pickle_data(f_name):
    """ Loads the pickle data """
    if not os.path.exists(f_name):
        raise Exception('Unable to find annotations picle file at %s. Aborting.'%(f_name))
    with open(f_name, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)

    return pickle_data

def project_3D_points(cam_mat, pts3D, is_OpenGL_coords=True):
    '''
    Function for projecting 3d points to 2d
    :param camMat: camera matrix
    :param pts3D: 3D points
    :param isOpenGLCoords: If True, hand/object along negative z-axis. If False hand/object along positive z-axis
    :return:
    '''
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if is_OpenGL_coords:
        pts3D = pts3D.dot(coord_change_mat.T)

    proj_pts = pts3D.dot(cam_mat.T)
    proj_pts = np.stack([proj_pts[:,0]/proj_pts[:,2], proj_pts[:,1]/proj_pts[:,2]],axis=1)

    assert len(proj_pts.shape) == 2

    return proj_pts

def read_RGB_img(base_dir, seq_name, file_id, split):
    """Read the RGB image in dataset"""
    if os.path.exists(os.path.join(base_dir, split, seq_name, 'rgb', file_id + '.png')):
        img_filename = os.path.join(base_dir, split, seq_name, 'rgb', file_id + '.png')
    else:
        img_filename = os.path.join(base_dir, split, seq_name, 'rgb', file_id + '.jpg')

    _assert_exist(img_filename)

    img = cv2.imread(img_filename)

    return img


def read_depth_img(base_dir, seq_name, file_id, split):
    """Read the depth image in dataset and decode it"""
    depth_filename = os.path.join(base_dir, split, seq_name, 'depth', file_id + '.png')

    _assert_exist(depth_filename)

    depth_scale = 0.00012498664727900177
    depth_img = cv2.imread(depth_filename)

    dpt = depth_img[:, :, 2] + depth_img[:, :, 1] * 256
    dpt = dpt * depth_scale

    return dpt

def read_annotation(base_dir, seq_name, file_id, split):
    meta_filename = os.path.join(base_dir, split, seq_name, 'meta', file_id + '.pkl')

    _assert_exist(meta_filename)

    pkl_data = load_pickle_data(meta_filename)

    return pkl_data


def load_faces():
    
    # Load right hand faces
    mano_layer = ManoLayer(mano_root='../mano_v1_2/models', use_pca=False, ncomps=6, flat_hand_mean=True)
    right_hand_faces = mano_layer.th_faces

    # Loading object faces
    obj_mesh = read_obj('../datasets/objects/mesh_1000/book.obj')
    obj_faces = obj_mesh.f

    # Load left hand faces
    mano_layer = ManoLayer(mano_root='../mano_v1_2/models', side='left', use_pca=False, ncomps=6, flat_hand_mean=True)
    left_hand_faces = mano_layer.th_faces
    
    return left_hand_faces, right_hand_faces, obj_faces


def save_mesh(outputs, filename, right_hand_faces, obj_faces, idx=0, texture=None, shape_dir='mesh', left_hand_faces=None):

    predicted_keypoints3d = outputs['mesh3d'][idx][:, :3]
    num_verts = outputs['mesh3d'][idx].shape[0]

    final_obj = filename.replace('visual_results', shape_dir).replace('.jpg', '').replace('.png', '')
    
    if outputs['mesh3d'][idx].shape[1] == 6:
        texture = outputs['mesh3d'][idx][:, 3:]
    else:
        texture = None

    # Disable object texture
    # texture[-1000:, :] = 0.5
    
    if num_verts == 2556:
        final_faces = np.concatenate((left_hand_faces, right_hand_faces + 778,  obj_faces + 778 * 2), axis = 0)
    elif num_verts == 1778:
        final_faces = np.concatenate((right_hand_faces, obj_faces + 778), axis = 0)
    else:
        final_faces = right_hand_faces

    write_obj(predicted_keypoints3d, final_faces, final_obj, texture)


def write_obj(verts, faces, filename, texture=None):
    """Saves and obj file using vertices and faces"""
    texture=None
    if texture is not None:
        alpha = np.ones((verts.shape[0], 1))
        v_color_matrix = np.append(texture, alpha, axis=1)
        m = pymeshlab.Mesh(verts, faces, v_color_matrix=v_color_matrix)
    else:
        m = pymeshlab.Mesh(verts, faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m, f'{filename}')
    ms.save_current_mesh(f'{filename}.obj', save_vertex_normal=True, save_vertex_color=True, save_polygonal=True)


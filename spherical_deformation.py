
"""
https://pytorch3d.org/tutorials/deform_source_mesh_to_target_mesh
"""
import pymeshlab
import os
import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
import trimesh
import argparse

parser = argparse.ArgumentParser(description='Spherical Deformation for YCB Objects')
parser.add_argument('--ycb_models_dir', type=str, help='YCB Models Directory')
parser.add_argument('--target_verts', type=int, help='Number of vertices in the target mesh')

args = parser.parse_args()

# Set the device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

YCBModelsDir = args.ycb_models_dir
target_verts = args.target_verts
os.makedirs("datasets/spheres", exist_ok=True)

ico_spheres = {
    642: 3,
    2556: 4
}
load_simplified_sphere = target_verts not in ico_spheres.keys()

def decimateMesh(verts, faces, targetverts=0, name="Mesh"):
    """Decimate the mesh to the target number of vertices using Quadric Edge Collapse Decimation (QECD)"""
    
    vertNotMatched = True
    threshold = 0.0001
    startValue = targetverts / verts.shape[0] - 5 * threshold
    while vertNotMatched:

        m = pymeshlab.Mesh(verts, faces)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(m, f"{name}")
        ms.meshing_decimation_quadric_edge_collapse(targetperc=startValue)
        startValue += threshold      
        decimated_graph = ms.current_mesh().vertex_matrix()
        decimated_faces = ms.current_mesh().face_matrix()

        if decimated_graph.shape[0] == targetverts:
            vertNotMatched = False
        elif decimated_graph.shape[0] > targetverts:
            startValue -= 10 * threshold
    
    ms.save_current_mesh(f"{name}.obj", save_vertex_normal=True, save_face_color=True, save_polygonal=True)
    return decimated_graph, decimated_faces

if load_simplified_sphere: 
    sphere = trimesh.creation.icosphere(subdivisions=ico_spheres[target_verts])
    sphere_vertices, sphere_faces = decimateMesh(sphere.vertices, sphere.faces, targetverts=target_verts, name=os.path.join('datasets/spheres',f'sphere_{target_verts}'))
    print(f'Simplified Sphere with {target_verts} Vertices Exported')
else:
    print(f'Icosphere with {target_verts} Vertices will be used')

for directory in os.listdir(YCBModelsDir):

    # Load the target mesh
    print(f'Deforming {directory} ...')
    trg_obj = os.path.join(YCBModelsDir, directory, 'textured_simple.obj')

    # We read the target 3D model using load_obj
    verts, faces, aux = load_obj(trg_obj)

    # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
    # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
    # For this tutorial, normals and textures are ignored.
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)

    # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0). 
    # (scale, center) will be used to bring the predicted mesh to its original center and scale
    # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale

    # We construct a Meshes structure for the target mesh
    trg_mesh = Meshes(verts=[verts], faces=[faces_idx])

    # We initialize the source shape to be a sphere of radius 1
    if load_simplified_sphere:
        src_obj = os.path.join('datasets/spheres/', f'sphere_{target_verts}.obj')
        verts, faces, aux = load_obj(src_obj)
        faces_idx = faces.verts_idx.to(device)
        verts = verts.to(device)
        src_mesh = Meshes(verts=[verts], faces=[faces_idx])
    
    else:
        src_mesh = ico_sphere(ico_spheres[target_verts], device)
    nVerts = src_mesh.verts_packed().shape[0]

    # We will learn to deform the source mesh by offsetting its vertices
    # The shape of the deform parameters is equal to the total number of vertices in src_mesh
    deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)

    # The optimizer
    optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)

    # Number of optimization steps
    Niter = 2000
    # # Weight for the chamfer loss
    w_chamfer = 1.0 
    # Weight for mesh edge loss
    w_edge = 1.0
    # Weight for mesh normal consistency
    w_normal = 0.01 
    # Weight for mesh laplacian smoothing
    w_laplacian = 0.1 

    # Plot period for the losses
    plot_period = 250
    loop = range(Niter)

    chamfer_losses = []
    laplacian_losses = []
    edge_losses = []
    normal_losses = []

    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()

        # Deform the mesh
        new_src_mesh = src_mesh.offset_verts(deform_verts)

        # We sample 5k points from the surface of each mesh 
        sample_trg = sample_points_from_meshes(trg_mesh, 5000)
        sample_src = sample_points_from_meshes(new_src_mesh, 5000)

        # We compare the two sets fosr pointclouds by computing (a) the chamfer loss
        loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)

        # and (b) the edge length of the predicted mesh
        loss_edge = mesh_edge_loss(new_src_mesh)

        # mesh normal consistency
        loss_normal = mesh_normal_consistency(new_src_mesh)

        # mesh laplacian smoothing
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")

        # Weighted sum of the losses
        loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian


        # Save the losses for plotting
        chamfer_losses.append(float(loss_chamfer.detach().cpu()))
        edge_losses.append(float(loss_edge.detach().cpu()))
        normal_losses.append(float(loss_normal.detach().cpu()))
        laplacian_losses.append(float(loss_laplacian.detach().cpu()))

        # Optimization step
        loss.backward()
        optimizer.step()


    # Fetch the verts and faces of the final predicted mesh
    final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

    # Scale normalize back to the original target size
    final_verts = final_verts * scale + center

    # Store the predicted mesh using save_obj
    final_obj = os.path.join(YCBModelsDir, directory, f'morphed_sphere_{target_verts}.obj')
    print('Saving to ', final_obj)
    save_obj(final_obj, final_verts, final_faces)



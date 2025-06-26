import wandb
import logging
import torch
import os
import numpy as np
import plyfile
from sklearn.cluster import KMeans
import struct
import skimage.measure
import random
import time
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
model_params_subdir = "ModelParameters"
optimizer_params_subdir = "OptimizerParameters"
latent_codes_subdir = "LatentCodes"
shape_latent_codes_subdir = "ShapeLatentCodes"
adjust_latent_codes_subdir = "AdjustLatentCodes"

def reply_temp_time(src_time):
    mylist = [(0.1,0),(0.9,4),(0.7,3),(0.5,2),(0.3,1)]
    return random.choice(mylist)

def wandb_name_init(debug=False,obj_name=None,dataset_name = None):
    wandb_dir = "exp_res/wandb"
    wandb_name = "SDFDecoder_debug_backward_final_"+dataset_name+"_"+obj_name
    if debug:
        wandb.init(project="3Dtemporal",name=wandb_name,dir=wandb_dir,mode="disabled")# mode="disabled"
    else:
        wandb.init(project="3Dtemporal",name=wandb_name,dir=wandb_dir)
    return wandb#,wandb_name
class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass

def load_model_parameters(experiment_directory, checkpoint, decoder):

    filename = os.path.join(
        experiment_directory, model_params_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))
    data = torch.load(filename)
    decoder.load_state_dict(data["model_state_dict"])

    return data["epoch"]

def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]

def load_latent_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        get_latent_codes_dir(experiment_directory), filename + ".pth"
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])
    return data["epoch"]
def load_shape_latent_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        get_shape_latent_codes_dir(experiment_directory), filename + ".pth"
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["shape_latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["shape_latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["shape_latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["shape_latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["shape_latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["shape_latent_codes"])
    return data["epoch"]

def load_adjust_latent_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        get_adjust_latent_codes_dir(experiment_directory), filename+ ".pth"
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["adjust_latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["adjust_latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["adjust_latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["adjust_latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["adjust_latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["adjust_latent_codes"])
    return data["epoch"]
class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value
    
class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length
def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules
def add_common_args(arg_parser):
    arg_parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed",
    )
    arg_parser.add_argument(
        "--log",
        dest="logfile",
        default=None,
        help="If set, the log will be saved using the specified filename.",
    )
def configure_logging(args):
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("DeepSdf - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if args.logfile is not None:
        file_logger_handler = logging.FileHandler(args.logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)

def get_model_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, model_params_subdir)
    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)
    return dir

def get_optimizer_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, optimizer_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir
#get_adjust_latent_codes_dir
def get_adjust_latent_codes_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, adjust_latent_codes_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir
def get_shape_latent_codes_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, shape_latent_codes_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir

def get_latent_codes_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, latent_codes_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir
def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )
def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )

def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )
def save_shape_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = get_shape_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "shape_latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )
def save_adjust_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = get_adjust_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "adjust_latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )
def save_latest(epoch,experiment_directory,decoder,optimizer_all,start_end_vecs):
    save_model(experiment_directory, "latest.pth", decoder, epoch)
    save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
    save_latent_vectors(experiment_directory, "latest.pth", start_end_vecs, epoch)
#ajust,velo_int,start_end_vecs
def save_adjust_checkpoints(epoch,experiment_directory,ajust,velo_int,start_end_vecs,adjust_start_end_vecs):
    model = torch.nn.ModuleList()
    model.append(ajust)
    model.append(velo_int)
    save_model(experiment_directory, str(epoch) + ".pth", model, epoch)
    save_latent_vectors(experiment_directory, str(epoch) + ".pth", start_end_vecs, epoch)
    save_adjust_latent_vectors(experiment_directory, str(epoch) + ".pth", adjust_start_end_vecs, epoch)
    #save_shape_latent_vectors(experiment_directory, str(epoch) + ".pth", shape_start_end_vecs, epoch)
def save_adjust_wo_checkpoints(epoch,experiment_directory,ajust,velo_int,start_end_vecs,adjust_start_end_vecs):
    model = torch.nn.ModuleList()
    model.append(ajust)
    model.append(velo_int)
    save_model(experiment_directory, str(epoch) + ".pth", model, epoch)
    save_latent_vectors(experiment_directory, str(epoch) + ".pth", start_end_vecs, epoch)
    save_adjust_latent_vectors(experiment_directory, str(epoch) + ".pth", adjust_start_end_vecs, epoch)
def save_checkpoints(epoch,experiment_directory,decoder,optimizer_all,start_end_vecs):
    save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch)
    save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
    save_latent_vectors(experiment_directory, str(epoch) + ".pth", start_end_vecs, epoch)
    
def save_forward_ode_checkpoints(epoch,experiment_directory,NDP,velo_int,start_end_vecs):
    model = torch.nn.ModuleList()
    model.append(NDP)
    model.append(velo_int)
    save_model(experiment_directory, str(epoch) + ".pth",model, epoch)
    save_latent_vectors(experiment_directory, str(epoch) + ".pth", start_end_vecs, epoch)
def save_forward_grid_checkpoints(epoch,experiment_directory,NDP,velo_int):
    model = torch.nn.ModuleList()
    model.append(NDP)
    model.append(velo_int)
    save_model(experiment_directory, str(epoch) + ".pth",model, epoch)
def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default

def adjust_learning_rate(lr_schedules, optimizer, epoch):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)  
def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def create_mesh_wmw(decoder,latent_vec,filename,N=256,
                    sm_offset=None,sm_len=None,lag_mean=None,lag_scale=None,max_batch=32 ** 3,this_time=None):
    start = time.time()
    ply_filename = filename
    decoder.eval()
    voxel_size = (sm_len / (N - 1))[0]

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)
    
    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size[0]) + sm_offset[0]
    samples[:, 1] = (samples[:, 1] * voxel_size[1]) + sm_offset[1]
    samples[:, 2] = (samples[:, 2] * voxel_size[2]) + sm_offset[2]
    num_samples = N ** 3

    samples.requires_grad = False
    head = 0
    
    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
        samples[head : min(head + max_batch, num_samples), 3] = (
            decode_sdf(decoder, latent_vec, sample_subset,this_time)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)
    end = time.time()

    print("sampling takes: %f" % (end - start))
    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        sm_offset.numpy(),
        voxel_size.numpy(),
        ply_filename + ".ply",
        lag_mean,
        lag_scale,
    )
def decode_sdf(decoder, latent_vector, queries,this_time):
    num_samples = queries.shape[0]

    if latent_vector is None:
        if this_time!=None:
            sdf = decoder.evaluate(queries,this_time)
        else:
            sdf = decoder.evaluate(queries)
    else:
        sdf = decoder.evaluate(queries,latent_vector)
    return sdf

def create_mesh_colored_wmw(decoder,latent_vec,filename,N=256,
                    sm_offset=None,sm_len=None,lag_mean=None,lag_scale=None,max_batch=32 ** 3,this_time =None):
    start = time.time()
    ply_filename = filename
    decoder.eval()
    voxel_size = (sm_len / (N - 1))[0]

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)
    
    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size[0]) + sm_offset[0]
    samples[:, 1] = (samples[:, 1] * voxel_size[1]) + sm_offset[1]
    samples[:, 2] = (samples[:, 2] * voxel_size[2]) + sm_offset[2]
    num_samples = N ** 3

    samples.requires_grad = False
    head = 0
    
    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
        samples[head : min(head + max_batch, num_samples), 3] = (
            decode_sdf(decoder.sdf_decoder, latent_vec, sample_subset,this_time)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)
    #np.save('sdf_values_others.npy',sdf_values)
    end = time.time()

    print("sampling takes: %f" % (end - start))
    colored_wmw_convert_sdf_samples_to_ply(
        decoder.partseg,
        sdf_values.data.cpu(),
        sm_offset.numpy(),
        voxel_size.numpy(),
        ply_filename + ".ply",
        lag_mean,
        lag_scale,
    )
def colored_wmw_convert_sdf_samples_to_ply(
    partseg,
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.0, spacing=voxel_size)

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0] + 0.5*voxel_size[0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1] + 0.5*voxel_size[1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2] + 0.5*voxel_size[2]
    
    mesh_colors = get_mesh_color(partseg, mesh_points)
    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale.numpy()
    if offset is not None:
        mesh_points = mesh_points + offset.numpy()

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    colors_tuple = np.zeros((num_verts,), dtype=[("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])
    for i in range(0, num_verts):
            colors_tuple[i] = tuple(mesh_colors[i, :])

    verts_all = np.empty(num_verts,verts_tuple.dtype.descr + colors_tuple.dtype.descr)

    for prop in verts_tuple.dtype.names:
        verts_all[prop] = verts_tuple[prop]

    for prop in colors_tuple.dtype.names:
        verts_all[prop] = colors_tuple[prop]


    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_all, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces],text=True)
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
def write_pointcloud(filename,xyz_points,rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))
    
    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffBBB",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0],rgb_points[i,1],rgb_points[i,2])))
    fid.close()

def color_pc(v):
    rgb_list=np.array([[153,0,0],[255,204,204],[255,229,204],[255,0,0],
                       [255,128,0],[255,255,0],[128,255,0],[0,153,0],[153,153,0],
                       [102,255,179],[102,255,255],[0,76,153],[0,0,255],[178,102,255],
                       [153,0,153],[255,102,255],[255,102,178],[192,192,192],[255,255,255]],\
        dtype=np.int32)
    clustering  = KMeans(n_clusters=20, random_state=0, n_init="auto").fit(v)
    pc_colors = np.zeros_like(v)
    pc_colors = rgb_list[clustering.labels_]
    return pc_colors
def get_mesh_color(part_seg, mesh_points):
    part_id = part_seg.get_part_id(torch.tensor(mesh_points).cuda()).detach().cpu().numpy()
    rgb_list=np.array([[192,192,192],[153,0,0],[255,204,204],[255,229,204],[255,0,0],
                    [255,128,0],[255,255,0],[128,255,0],[0,153,0],[153,153,0],
                    [102,255,179],[102,255,255],[0,76,153],[0,0,255],[178,102,255],
                    [153,0,153],[255,102,255],[255,102,178],[0,0,128],[255,255,255]],
                    dtype=np.int32)
    pc_colors = np.zeros_like(mesh_points)
    pc_colors = rgb_list[part_id]
    return pc_colors

def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.0, spacing=voxel_size)

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0] + 0.5*voxel_size[0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1] + 0.5*voxel_size[1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2] + 0.5*voxel_size[2]
      
    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale.numpy()
    if offset is not None:
        mesh_points = mesh_points + offset.numpy()

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
import torch
from typing import Union
import wandb
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds
import torch.autograd as ag
import torch.nn as nn
from .part_loss import corresponding_points_alignment_loss

class GenLoss(nn.Module):
    def __init__(self,):
        super().__init__()
class ReconLoss(GenLoss):
    def __init__(self,**kwargs):
        super().__init__()
        self.latent_reg_weight = kwargs['latent_reg_weight'] #0.001
        self.grad_loss_weight = kwargs['grad_loss_weight'] #0.1
        self.grad_on_surface_weight = kwargs['grad_on_surface_weight'] #0.1
        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
    
    def df(self,x,wrt):
        M = x.shape[0]
        return ag.grad(x.flatten(), wrt,
                        grad_outputs=torch.ones(M).to(x.dtype).to(wrt.device),create_graph=True, retain_graph=True)[0]
    def forward(self,outputs):
        # compute the eilonal_loss
        uniform_grad = self.df(outputs["uniform_preds"],outputs["uniform_pts"])
        eikonal_loss_1 = (torch.square(torch.norm(uniform_grad,dim=1) - 1)).mean()
        eikonal_loss_1 = eikonal_loss_1 * self.grad_loss_weight

        near_grad = self.df(outputs["near_preds"],outputs["near_pts"])
        eikonal_loss_2 = (torch.square(torch.norm(near_grad,dim=1) - 1)).mean()
        eikonal_loss_2 = eikonal_loss_2 * self.grad_loss_weight


        #reg_loss = self.latent_reg_weight * (outputs["latent_code"]**2).mean()
      
        # surface norm loss
        grad_surface = self.df(outputs["mns_preds"],outputs["pts"])

        surface_norm_loss = self.l1_loss(grad_surface,outputs['gt_nms'])
        surface_norm_loss = self.grad_on_surface_weight * surface_norm_loss
        # network gts loss
        gts_loss = (torch.abs(outputs['mns_preds'])).mean()
        #psuedo gt_dist:
        psuedo_dist = (knn_points(outputs["near_pts"].unsqueeze(0),
                    outputs["pts"].unsqueeze(0),K=1).dists[0][:,-1])**0.5
        
        recon_loss =  torch.min(torch.abs(psuedo_dist - outputs["near_preds"]),
                                torch.abs(psuedo_dist + outputs["near_preds"])).mean()
        
        # grad loss for near surface
        psuedo_dist_dxyz = self.df(psuedo_dist,outputs["near_pts"]).detach()
        
        
        grad_loss =  torch.min(torch.norm(torch.abs(psuedo_dist_dxyz - near_grad),dim=1),
                                    torch.norm(torch.abs(psuedo_dist_dxyz + near_grad),dim=1)).mean()
        

        grad_loss = self.grad_on_surface_weight * grad_loss
        # total loss computing
        total_loss = 0.5*(eikonal_loss_1+eikonal_loss_2) + 0.5*(recon_loss+gts_loss)+ 0.5*(surface_norm_loss+grad_loss)
        #total_loss = reg_loss + 0.5*(eikonal_loss_1+eikonal_loss_2) + \
        #0.5*(recon_loss+gts_loss)+ 0.5*(surface_norm_loss+grad_loss)
        return total_loss

class ReconLoss_debug(GenLoss):
    def __init__(self):
        super().__init__()
        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.identity = torch.eye(3).reshape((1, 3, 3)).cuda()
    def df(self,x,wrt):
        M = x.shape[0]
        return ag.grad(x.flatten(), wrt,
                        grad_outputs=torch.ones(M).to(x.dtype).to(wrt.device),create_graph=True)[0]
    def probability_l1_loss(self,input, target,condition=None, reduction='mean'):
        # Calculate the absolute difference
        if condition != None:
            loss = condition*torch.norm(torch.abs(input - target),dim=1,keepdim=True)
        else:
            loss = torch.norm(torch.abs(input - target),dim=1,keepdim=True)
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def forward(self,outputs,src_prob=None):
        # compute the eilonal_loss
        losses = {}
        ###################################################################

        uniform_grad = self.df(outputs["uniform_preds"],outputs["uniform_pts"])
        #del outputs["uniform_pts"].grad

        eikonal_loss_1 = (torch.square(torch.norm(uniform_grad,dim=1) - 1)).mean()
        #eikonal_loss_1 = eikonal_loss_1 * self.grad_loss_weight

        near_grad = self.df(outputs["near_preds"],outputs["near_pts"])

        surface_alig_loss =corresponding_points_alignment_loss(outputs['mns_final'].unsqueeze(0),
                                                    outputs['pts'].unsqueeze(0),
                                                    (outputs['mns_prob'].transpose(-1,-2)).unsqueeze(0),
                                                    eps_backward = 1e-7).sum(dim=-1)/len(outputs['pts'])
        

        eikonal_loss_2 = (torch.square(torch.norm(near_grad,dim=1) - 1)).mean()

        # surface norm loss
        grad_surface = self.df(outputs["mns_preds"],outputs["pts"])
        #del outputs["pts"].grad
        surface_norm_loss = self.probability_l1_loss(grad_surface,outputs['gt_nms'],condition=src_prob)
        #surface_norm_loss = self.grad_on_surface_weight * surface_norm_loss
        # network gts loss
        if src_prob==None:
            gts_loss = (torch.abs(outputs['mns_preds'])).mean()
            psuedo_dist = ((knn_points(outputs["near_pts"].unsqueeze(0),
                        outputs["pts"].unsqueeze(0),K=1).dists[0][:,-1])**0.5).unsqueeze(-1)
            recon_loss =  (torch.min(torch.abs(psuedo_dist - outputs["near_preds"]),
                        torch.abs(psuedo_dist + outputs["near_preds"]))).mean()
            # grad loss for near surface
            psuedo_dist_dxyz = self.df(psuedo_dist,outputs["near_pts"]).detach()
            align_loss = surface_alig_loss.item()
            grad_loss =  (torch.min(torch.norm(torch.abs(psuedo_dist_dxyz - near_grad),dim=1,keepdim=True),
                                    torch.norm(torch.abs(psuedo_dist_dxyz + near_grad),dim=1,keepdim=True))).mean()
        else:

            gts_loss = (src_prob * torch.abs(outputs['mns_preds'])).mean()
            psuedo_dist = ((knn_points(outputs["near_pts"].unsqueeze(0),
                        outputs["pts"].unsqueeze(0),K=1).dists[0][:,-1])**0.5).unsqueeze(-1)
            src_prob_exps = torch.cat([src_prob,src_prob],dim=0)
            
            recon_loss =  (src_prob_exps*torch.min(torch.abs(psuedo_dist - outputs["near_preds"]),
                                    torch.abs(psuedo_dist + outputs["near_preds"]))).mean()
            # grad loss for near surface
            psuedo_dist_dxyz = self.df(psuedo_dist,outputs["near_pts"]).detach()
            #del outputs["near_pts"].grad
            align_loss = surface_alig_loss.item()
            grad_loss =  (src_prob_exps*torch.min(torch.norm(torch.abs(psuedo_dist_dxyz - near_grad),dim=1,keepdim=True),
                                        torch.norm(torch.abs(psuedo_dist_dxyz + near_grad),dim=1,keepdim=True))).mean()

        losses["eikonal_uniform"] = eikonal_loss_1
        losses["eikonal_near"] = eikonal_loss_2
        losses['surface_norm'] = surface_norm_loss
        losses['surface_pts'] = gts_loss
        losses['near_pts'] = recon_loss
        losses['near_norm'] = grad_loss
        #losses['latent_loss'] = reg_loss
        losses['align_loss'] = align_loss
        return losses
    
def _validate_chamfer_reduction_inputs(
        batch_reduction: Union[str, None], point_reduction: str
):
    """Check the requested reductions are valid.
    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')

def _handle_pointcloud_input(
        points: Union[torch.Tensor, Pointclouds],
        lengths: Union[torch.Tensor, None],
        normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None and (
                lengths.ndim != 1 or lengths.shape[0] != X.shape[0]
        ):
            raise ValueError("Expected lengths to be of shape (N,)")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals
def compute_truncated_chamfer_distance(
        x,
        y,
        x_prob=None,
        y_prob=None,
        x_lengths=None,
        y_lengths=None,
        x_normals=None,
        y_normals=None,
        weights=None,
        trunc=0.2,
        batch_reduction: Union[str, None] = "mean",
        point_reduction: str = "mean",
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
            torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
            torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)


    # truncation
    x_mask[cham_x >= trunc] = True
    y_mask[cham_y >= trunc] = True
    cham_x[x_mask] = 0.0
    cham_y[y_mask] = 0.0


    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    # Apply point reduction
      
    #cham_x = cham_x.sum(1)  # (N,)
    #cham_y = cham_y.sum(1)  # (N,)

    # use l1 norm, more robust to partial case
    keep_list = torch.sqrt(cham_x)
    keep_list = keep_list/keep_list.sum(1)
    if (x_prob==None) and (y_prob==None):
        cham_x = torch.sqrt(cham_x).sum(1)  # (N,)
        cham_y = torch.sqrt(cham_y).sum(1)  # (N,)
    else:
        cham_x = torch.sqrt((x_prob.T)*cham_x).sum(1) 
        cham_y = torch.sqrt((y_prob.T)*cham_y).sum(1) 
    if return_normals:
        if x_prob==None and y_prob==None:
            cham_norm_x = cham_norm_x.sum(1)  # (N,)
            cham_norm_y = cham_norm_y.sum(1)  # (N,)
        else:
            cham_norm_x = ((x_prob.T)*cham_norm_x).sum(1)
            cham_norm_y = ((y_prob.T)*cham_norm_y).sum(1)

    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths
    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div
    cham_dist = cham_x + cham_y
    if return_normals:
        cham_normals = cham_norm_x + cham_norm_y
        return cham_dist,cham_normals,keep_list.detach()
    else:
        return cham_dist,keep_list.detach()

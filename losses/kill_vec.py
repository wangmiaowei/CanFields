import torch
def kill_loss(total_grads,near_cost):
    if near_cost>8:
        near_cost=8
    total_grads_1 = torch.cat(total_grads[:near_cost],dim=0) #near pts
    total_grads_2 = torch.cat(total_grads[near_cost:],dim=0) #uniform pts

    kl_loss_1 = torch.norm(total_grads_1 + total_grads_1.transpose(2,3),
                            dim=[2,3]).mean(dim=1).mean(dim=0)
    kl_loss_2 = torch.norm(total_grads_2 + total_grads_2.transpose(2,3),
                            dim=[2,3]).mean(dim=1).mean(dim=0)
    
    l2_ve_1 = torch.norm(total_grads_1,dim=[2,3]).mean(dim=1).mean(dim=0)
    l2_ve_2 = torch.norm(total_grads_2,dim=[2,3]).mean(dim=1).mean(dim=0)

    kl_loss_2 = torch.norm(total_grads_2 + total_grads_2.transpose(2,3),
                            dim=[2,3]).mean(dim=1).mean(dim=0)
    kl_loss_total = 0.7*kl_loss_1+0.3*kl_loss_2
    l2_ve_total = 0.7*l2_ve_1+0.3*l2_ve_2
    return kl_loss_total, l2_ve_total
def compute_fit_total_loss(recon_losses,grad_loss_weight,
                           grad_on_surface_weight,latent_reg_weight):
    eikonal_uniform_loss  = recon_losses["eikonal_uniform"]
    eikonal_near_loss = recon_losses["eikonal_near"]
    surface_norm_loss = recon_losses["surface_norm"]
    surface_pts_loss = recon_losses["surface_pts"]
    near_pts_loss =  recon_losses["near_pts"]
    near_norm_loss = recon_losses["near_norm"]
    latent_reg_loss = recon_losses["latent_loss"]
    fit_loss = 0.5*grad_loss_weight * (eikonal_uniform_loss+eikonal_near_loss) +\
        0.5*(surface_pts_loss+near_pts_loss) +\
              0.5*grad_on_surface_weight*(surface_norm_loss+near_norm_loss)+\
                latent_reg_weight*latent_reg_loss
    return fit_loss
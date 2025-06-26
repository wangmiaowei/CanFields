import torch
import numpy as np

def slerp(v0: torch.FloatTensor, v1: torch.FloatTensor, t: float, DOT_THRESHOLD=0.9995):
  '''
  Spherical linear interpolation
  Args:
    v0: Starting vector
    v1: Final vector
    t: Float value between 0.0 and 1.0
    DOT_THRESHOLD: Threshold for considering the two vectors as
                            colinear. Not recommended to alter this.
  Returns:
      Interpolation vector between v0 and v1
  '''
  assert v0.shape == v1.shape, "shapes of v0 and v1 must match"

  # Normalize the vectors to get the directions and angles
  v0_norm = torch.linalg.norm(v0, dim=-1)
  v1_norm = torch.linalg.norm(v1, dim=-1)

  v0_normed = v0 / v0_norm.unsqueeze(-1)
  v1_normed = v1 / v1_norm.unsqueeze(-1)

  # Dot product with the normalized vectors
  dot = (v0_normed * v1_normed).sum(-1)
  dot_mag = dot.abs()

  # if dp is NaN, it's because the v0 or v1 row was filled with 0s
  # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
  gotta_lerp = dot_mag.isnan() | (dot_mag > DOT_THRESHOLD)
  can_slerp = ~gotta_lerp

  t_batch_dim_count = max(0, t.dim()-v0.dim()) if isinstance(t, torch.Tensor) else 0
  t_batch_dims = t.shape[:t_batch_dim_count] if isinstance(t, torch.Tensor) else torch.Size([])
  out = torch.zeros_like(v0.expand(*t_batch_dims, *[-1]*v0.dim()))

  # if no elements are lerpable, our vectors become 0-dimensional, preventing broadcasting
  if gotta_lerp.any():
    lerped  = torch.lerp(v0, v1, t)

    out = lerped.where(gotta_lerp.unsqueeze(-1), out)

  # if no elements are slerpable, our vectors become 0-dimensional, preventing broadcasting
  if can_slerp.any():

    # Calculate initial angle between v0 and v1
    theta_0 = dot.arccos().unsqueeze(-1)
    sin_theta_0 = theta_0.sin()
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = theta_t.sin()
    # Finish the slerp algorithm
    s0 = (theta_0 - theta_t).sin() / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    slerped = s0 * v0 + s1 * v1

    out  = slerped.where(can_slerp.unsqueeze(-1), out)
  return out


def fit_latent(latent,t):
    whole_seg = None
    z1= latent[0]
    z2 = latent[1]
    z1_n = torch.norm(z1) + 1e-9
    z2_n = torch.norm(z2) + 1e-9
    slp_int = slerp(z1/z1_n,z2/z2_n,t)
    zt = ((1-t)*z1_n + t*z2_n) * slp_int
    return zt
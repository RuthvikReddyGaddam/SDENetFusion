import torch
import torch.nn as nn

class lbp_conv(nn.Module):
  def __init__(self):
    super().__init__()
    # self.lbp_weights = torch.tensor([1,2,4,8,16,32,64,128], dtype=torch.float32).view(1,8,1,1)
    self.kernels = torch.tensor([
      [[ 1, 0, 0],
      [ 0,-1, 0],
      [ 0, 0, 0]],

      [[ 0, 1, 0],
      [ 0,-1, 0],
      [ 0, 0, 0]],

      [[ 0, 0, 1],
      [ 0,-1, 0],
      [ 0, 0, 0]],

      [[ 0, 0, 0],
      [ 0,-1, 1],
      [ 0, 0, 0]],

      [[ 0, 0, 0],
      [ 0,-1, 0],
      [ 0, 0, 1]],

      [[ 0, 0, 0],
      [ 0,-1, 0],
      [ 0, 1, 0]],

      [[ 0, 0, 0],
      [ 0,-1, 0],
      [ 1, 0, 0]],

      [[ 0, 0, 0],
      [ 1,-1, 0],
      [ 0, 0, 0]],
       ], dtype=torch.float32)
    self.kernels = self.kernels.view(8,1,3,3)
    self.register_buffer("lbp_kernels", self.kernels)

  def forward(self, x):
    B, C, H, W = x.shape
    kernels = self.lbp_kernels.repeat(C, 1, 1, 1)

    lbp_convs = torch.nn.functional.conv2d(x, kernels, padding=1, groups=C)
    # lbp_convs = lbp_convs * self.lbp_weights
    # lbp_convs = torch.sum(lbp_convs, dim=1, keepdim=True)
    return lbp_convs
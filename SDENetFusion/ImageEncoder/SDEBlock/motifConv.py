import torch
from torch import nn
class motif_conv(nn.Module):
  def __init__(self):
    super().__init__()
    self.kernels, self.motif_splits = self.create_peano_motifs()
    self.register_buffer("motif_kernels", self.kernels)

  # function to create motif kernels
  def create_motifs(self, a,b):
    kernel = torch.zeros((1,1,2,2), dtype=torch.float32)
    ax, ay = divmod(a,2)
    bx, by = divmod(b,2)
    kernel[0,0,ax,ay] = 1.0
    kernel[0,0,bx,by] = -1.0
    return kernel

  def get_motifs(self, pattern):
    kernels = torch.empty((0,1,2,2), dtype=torch.float32)
    for i in range(len(pattern)-1):
      kernels = torch.cat([kernels, self.create_motifs(pattern[i], pattern[i+1])], dim=0)
    return kernels

  def create_peano_motifs(self):
    peano_motifs = [
    [0,1,2,3], [0,1,3,2], [0,2,1,3],
    [0,2,3,1], [0,3,1,2], [0,3,2,1],
    [3,2,1,0], [3,2,0,1], [3,1,2,0],
    [3,1,0,2], [3,0,2,1], [3,0,1,2]
    ]
    motif_kernels = []
    motif_splits = []
    idx = 0

    for motif in peano_motifs:
      motif_kernels.append(self.get_motifs(motif))
      motif_splits.append([idx, idx+1, idx+2])
      idx+=3
    motif_kernels = torch.cat(motif_kernels, dim=0)
    return motif_kernels, motif_splits

  def forward(self, x):
    B, C, H, W = x.shape
  
    x = torch.nn.functional.pad(x, (0,1,0,1))
    kernels = self.motif_kernels.repeat(C, 1, 1, 1)
    motif_convs = torch.nn.functional.conv2d(x, kernels, groups=C)

    motif_convs = motif_convs.view(B, C, 36, H, W)
    motif_maps = []
    for s0, s1, s2 in self.motif_splits:
        m = motif_convs[:, :, s0:s2+1, :, :].sum(dim=2, keepdim=True)
        motif_maps.append(m)

    motif_maps = torch.cat(motif_maps, dim=2)

    min_map = motif_maps.min(dim=2, keepdim=True).values
    med_map = motif_maps.median(dim=2, keepdim=True).values
    max_map = motif_maps.max(dim=2, keepdim=True).values

    motif_maps = torch.cat([motif_maps, min_map, med_map, max_map], dim=2)

    out = motif_maps.flatten(1, 2)
    return out
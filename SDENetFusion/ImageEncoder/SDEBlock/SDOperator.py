import torch
import torch.nn as nn
from SDENetFusion.ImageEncoder.SDEBlock.lbpConv import lbp_conv
from SDENetFusion.ImageEncoder.SDEBlock.motifConv import motif_conv

class SDOperator(nn.Module):
    def __init__(self, r=3, in_channels=3, dec_in_channels=3, k_size=3, padding=1, stride=1, input=True):
        super().__init__()
        self.input = input
        self.C_enc = in_channels
        self.C_dec = dec_in_channels
        self.r = r
        self.C_sq = self.C_enc // self.r
        
        self.conv1 = nn.Conv2d(in_channels=self.C_enc, out_channels=self.C_sq, kernel_size=k_size, padding=padding, stride=stride)
        
        self.lbp_conv = lbp_conv()
        self.motif_conv = motif_conv()

        if self.input == False:
            self.upsample = nn.ConvTranspose2d(in_channels=self.C_dec, out_channels=self.C_dec, kernel_size=2, padding=0, stride=2)
        
    def forward(self, U_enc, U_dec):
        U_sq = self.conv1(U_enc)
        U_lbp = self.lbp_conv(U_sq)
        U_motif = self.motif_conv(U_sq)
        U_de = torch.cat([U_lbp, U_motif], dim=1)
        if not self.input:
            U_dec = self.upsample(U_dec)
        U_sd = torch.cat([U_de, U_dec], dim=1)
     
        return U_sd
        
        
         
        
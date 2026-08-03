import torch
import torch.nn as nn
from SDENetFusion.ImageEncoder.SDEBlock.SDOperator import SDOperator
from SDENetFusion.ImageEncoder.SDEBlock.SEOperator import SEOperator

class SDEBlock(nn.Module):
    def __init__(self, in_channels=3, dec_in_channels=3, r=3, k_size=3, padding=1, stride=1, input=True):
        super().__init__()
        self.C_enc = in_channels
        self.C_dec = dec_in_channels
        self.r = r
        self.C_sd = self.calculate_C_sd()
        self.sd_operator = SDOperator(r=self.r, in_channels=in_channels, dec_in_channels=dec_in_channels, k_size=k_size, padding=padding, stride=stride, input=input)
        self.se_operator = SEOperator(C_sd = self.C_sd)

    def calculate_C_sd(self):
        C_sq = self.C_enc // self.r
        C_lbp = 8 * C_sq
        C_motif = 15 * C_sq
        return C_lbp + C_motif + self.C_dec
    
    def forward(self, U_enc, U_dec):
        U_sd = self.sd_operator(U_enc, U_dec)
        U_ex = self.se_operator(U_sd)
        return U_ex
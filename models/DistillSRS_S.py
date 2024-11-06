'''
    This is the student model for DistillSRS-T, they share the same backbone, and the student model aligns with the teacher model by minimizing the KL divergence between their logits and hidden states, and the attention scores in the self-attention with marker module.
'''

import torch
import torch.nn as nn
from models.SASRec import SASRec
from models.DualLLMSRS import DualLLMSASRec

class Distill_SASRec_(DualLLMSASRec):

    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)

        self.alpha = args.alpha
        self.ranking_align_loss = torch.nn.BCEWithLogitsLoss()
        self.hidden_align_loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.attention_align_loss = torch.nn.KLDivLoss(reduction='batchmean')













# here put the import lib
import os
import copy
import random
import numpy as np
from torch.utils.data import Dataset
from utils.utils import random_neq
import pickle


class SeqDataset_att(Dataset):
    '''The train dataset for Sequential recommendation'''

    def __init__(self, data, attention_marker, item_num, max_len, neg_num=1):
        
        super().__init__()
        self.data = data
        self.attention_marker = attention_marker
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.var_name = ["seq", "pos", "neg", "positions", "marked_seq"]


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        inter = self.data[index]
        marked_seq = self.attention_marker[index]
        non_neg = copy.deepcopy(inter)
        pos = inter[-1]
        neg = []
        for _ in range(self.neg_num):
            per_neg = random_neq(1, self.item_num+1, non_neg)
            neg.append(per_neg)
            non_neg.append(per_neg)
        neg = np.array(neg)
        #neg = random_neq(1, self.item_num+1, inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, pos, neg, positions, marked_seq
    


class Seq2SeqDataset_att(Dataset):
    '''The train dataset for Sequential recommendation with seq-to-seq loss'''

    def __init__(self, args, data, attention_marker, item_num, max_len, neg_num=1):
        
        super().__init__()
        self.data = data
        self.attention_marker = attention_marker
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.aug_seq = args.aug_seq
        self.aug_seq_len = args.aug_seq_len
        self.var_name = ["seq", "pos", "neg", "positions", "marked_seq"]


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        inter = self.data[index]
        marked_seq = self.attention_marker[index]
        non_neg = copy.deepcopy(inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)

        nxt = inter[-1]
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, self.item_num+1, non_neg)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        if self.aug_seq:
            seq_len = len(inter)
            pos[:- (seq_len - self.aug_seq_len) + 1] = 0
            neg[:- (seq_len - self.aug_seq_len) + 1] = 0
            marked_seq[:- (seq_len - self.aug_seq_len) + 1] = 0
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, pos, neg, positions, marked_seq




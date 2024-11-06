# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs
    


class Contrastive_Loss2(nn.Module):

    def __init__(self, tau=1) -> None:
        super().__init__()

        self.temperature = tau


    def forward(self, X, Y):
        
        logits = (X @ Y.T) / self.temperature
        X_similarity = Y @ Y.T
        Y_similarity = X @ X.T
        targets = F.softmax(
            (X_similarity + Y_similarity) / 2 * self.temperature, dim=-1
        )
        X_loss = self.cross_entropy(logits, targets, reduction='none')
        Y_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (Y_loss + X_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    

    def cross_entropy(self, preds, targets, reduction='none'):

        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
    


class CalculateAttention(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, Q, K, V, mask):

        attention = torch.matmul(Q,torch.transpose(K, -1, -2))
        # use mask
        attention = attention.masked_fill_(mask, -1e9)
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention,V)
        return attention



class Multi_CrossAttention(nn.Module):
    """
    forward时，第一个参数用于计算query，第二个参数用于计算key和value
    """
    def __init__(self,hidden_size,all_head_size,head_num):
        super().__init__()
        self.hidden_size    = hidden_size       # 输入维度
        self.all_head_size  = all_head_size     # 输出维度
        self.num_heads      = head_num          # 注意头的数量
        self.h_size         = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)


    def print(self):
        print(self.hidden_size,self.all_head_size)
        print(self.linear_k,self.linear_q,self.linear_v)
    

    def forward(self,x,y,log_seqs):
        """
        cross-attention: x,y是两个模型的隐藏层，将x作为q的输入，y作为k和v的输入
        """

        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # attention_mask = attention_mask.eq(0)
        attention_mask = (log_seqs == 0).unsqueeze(1).repeat(1, log_seqs.size(1), 1).unsqueeze(1)

        attention = CalculateAttention()(q_s,k_s,v_s,attention_mask)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        
        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        return output


class SelfAttention_with_marker(nn.Module):
    '''
    To use this class, you need to provide the marked_positions tensor, which indicates the positions of the marked elements in the input sequence.
    marked_positions = torch.tensor([[1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1]])
    self_attention = SelfAttention(hidden_size, all_head_size, head_num)
    output = self_attention(x, log_seqs, marked_positions) size: (batch_size, seq_len, hidden_size)
    '''
    def __init__(self, hidden_size, all_head_size, head_num, marker_dim=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.all_head_size = all_head_size
        self.num_heads = head_num
        self.h_size = all_head_size // head_num
        self.marker_dim = marker_dim

        assert all_head_size % head_num == 0

        # Marker embedding
        self.marker_embedding = nn.Embedding(2, marker_dim)
        
        # Linear projections
        self.linear_q = nn.Linear(hidden_size + marker_dim, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size + marker_dim, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size + marker_dim, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        self.norm = sqrt(all_head_size)

    def forward(self, x, log_seqs, seq_marker):
        batch_size = x.size(0)
        # size of seq_marker: [batch_size, seq_len], seq_marker[i][j] = 1 if j is the index of attention marker for user i
        # Generate marker embeddings
        marker_emb = self.marker_embedding(seq_marker.long())  # size of marker_emb: [batch_size, seq_len, marker_dim]

        # Concatenate input with marker embeddings
        x_marked = torch.cat([x, marker_emb], dim=-1) # dim = -1 means concatenate along the last dimension

        # Project input
        q_s = self.linear_q(x_marked).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        k_s = self.linear_k(x_marked).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        v_s = self.linear_v(x_marked).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # Create attention mask
        attention_mask = (log_seqs == 0).unsqueeze(1).repeat(1, log_seqs.size(1), 1).unsqueeze(1)

        # Calculate attention
        attention = CalculateAttention()(q_s, k_s, v_s, attention_mask)
        
        # Reshape and project output
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        output = self.linear_output(attention)

        return output

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, all_head_size, head_num):
        super().__init__()
        self.hidden_size = hidden_size       # input dimension
        self.all_head_size = all_head_size   # output dimension
        self.num_heads = head_num            # number of attention heads
        self.h_size = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q, W_K, W_V (hidden_size, all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)

    def forward(self, x, log_seqs):
        """
        self-attention: x is used as input for query, key, and value
        """
        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s, k_s, v_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        k_s = self.linear_k(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        v_s = self.linear_v(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # attention_mask
        attention_mask = (log_seqs == 0).unsqueeze(1).repeat(1, log_seqs.size(1), 1).unsqueeze(1)

        attention = CalculateAttention()(q_s, k_s, v_s, attention_mask)
        # attention: [batch_size, seq_length, num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        
        # output: [batch_size, seq_length, hidden_size]
        output = self.linear_output(attention)

        return output
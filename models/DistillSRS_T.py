# here put the import lib
import torch
import torch.nn as nn
from models.DualLLMSRS import DualLLMSASRec, DualLLMGRU4Rec, DualLLMBert4Rec
from models.utils import Contrastive_Loss2
from models.utils import SelfAttention_with_marker
import json

class Distill_SASRec_T(DualLLMSASRec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        # in our distill-srs-Teacher model, the embedding alignment is conducted through previous fusion, 
        # and the attention alignment is conducted through the self-attention module with markers, 
        # and the ranking alignment refers to https://github.com/istarryn/DLLM2Rec
        self.alpha = args.alpha
        self.item_num = item_num
        self.device = device
        #self.self_attention_module = SelfAttention_with_marker(args.hidden_size, args.hidden_size, 2, marker_dim=args.marker_dim)
        
        # top-k  items ranking loss
          # implemented in trainer
       #self.id2llm = SelfAttention_with_marker(args.hidden_size, args.hidden_size, 2)
        #self.user_sim_func = args.user_sim_func
        #self.item_reg = args.item_reg

        #if self.user_sim_func == "cl":
        #    self.align = Contrastive_Loss2()
        #   elif self.user_sim_func == "kd":
        #    self.align = nn.MSELoss()
        #else:
        #    raise ValueError

        #   self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        # self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        #   if self.item_reg:
        #    self.beta = args.beta
        #    self.reg = Contrastive_Loss2()

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                marked_seq,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, marked_seq, **kwargs)  # get the original loss
        
        log_feats = self.log2feats_with_attention_marker(seq, positions, marked_seq)[:, -1, :] # size: (batch_size, hidden_size)
        # pos: size: (batch_size, seq_len)
        # neg: size: (batch_size, seq_len, num_neg)
        all_items_logits = log_feats * self._get_embedding(torch.arange(self.item_num).to(self.device))

        # find the top-k items from the log_feats * self._get_embedding(all_items)
        #all_items = torch.arange(self.item_num).to(self.dev)
        #all_items_embs = self._get_embedding(all_items)
        #all_items_logits = log_feats * all_items_embs
        # 

        '''
        # find the top-k items from the log_feats * self._get_embedding(all_items)
        all_items = torch.arange(self.item_num).to(self.dev)
        all_items_embs = self._get_embedding(all_items)
        all_items_logits = log_feats * all_items_embs
        topk_items_logits, topk_items_indices = torch.topk(all_items_logits, k=self.topk, dim=-1)
      
        # calculate logits of all items
        all_items = torch.arange(self.item_num).to(self.dev)
        all_items_embs = self._get_embedding(all_items)
        all_items_logits = (log_feats.unsqueeze(1) * all_items_embs.unsqueeze(0)).sum(dim=-1)
        # all_items_logits shape: (batch_size, num_items)

        # Create a mask for non-padded positions
        mask = (pos != 0)  # shape: (batch_size,)

        # Set logits for padded positions to a very low value
        min_value = torch.finfo(all_items_logits.dtype).min
        all_items_logits = all_items_logits.masked_fill(~mask.unsqueeze(1), min_value)

        # Get top-k items, ignoring padded positions
        topk_items_logits, topk_items_indices = torch.topk(all_items_logits, k=self.topk, dim=-1)
        # topk_items_logits and topk_items_indices shape: (batch_size, topk)

        # Load LLM picked top-k items
        llm_topk_items = json.load(open(".data/"+self.args.dataset+"/parsed/llm_topk_items.json", "r"))
        llm_topk_items = torch.tensor(llm_topk_items).to(self.dev)
        llm_topk_items = llm_topk_items.view(-1, self.topk)
        # llm_topk_items shape: (batch_size, topk)

        # Calculate the ranking loss only for non-padded positions
        # llm_top_items as labels in BCELoss, only considering the top-k items
        ranking_labels = (topk_items_indices == llm_topk_items).float()

        # Add ranking loss to the total loss
        loss += self.alpha * ranking_loss

        #  sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq.shape[1]), kwargs["sim_positions"].view(-1, seq.shape[1])
        # sim_num = kwargs["sim_seq"].shape[1]
        # sim_log_feats = self.log2feats(sim_seq, sim_positions)[:, -1, :]    # (bs*sim_num, hidden_size)
        # sim_log_feats = sim_log_feats.detach().view(seq.shape[0], sim_num, -1)  # (bs, sim_num, hidden_size)
        # sim_log_feats = torch.mean(sim_log_feats, dim=1)

        #if self.user_sim_func == "cl":
        #    # align_loss = self.align(self.projector1(log_feats), self.projector2(sim_log_feats))
        #    align_loss = self.align(log_feats, sim_log_feats)
        #elif self.user_sim_func == "kd":
        #    align_loss = self.align(log_feats, sim_log_feats)

        #if self.item_reg:
        #    unfold_item_id = torch.masked_select(seq, seq>0)
        #    llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))
        #    id_item_emb = self.id_item_emb(unfold_item_id)
        #    reg_loss = self.reg(llm_item_emb, id_item_emb)
        #     loss += self.beta * reg_loss

        #loss += self.alpha * align_loss
    '''
        # ranking loss will be implemented in trainer
        return loss, all_items_logits
    

import torch
import torch.nn as nn
import math

class CoPE ( nn . Module ) :
    def __init__( self , npos_max , head_dim ):
        super () . __init__ ()
        self.npos_max = npos_max 
        # context的位置编码，参数可以更新调整。
        self.pos_emb = nn.parameter.Parameter (torch.zeros (1 , head_dim , npos_max ))

    def forward( self , query , attn_logits ):
         # compute positions
         gates = torch . sigmoid ( attn_logits )  
         # flip(-1)
         pos = gates . flip ( -1) . cumsum ( dim = -1) . flip ( -1)
         pos = pos . clamp (max = self . npos_max - 1)
         
         # interpolate from integer positions
         pos_ceil = pos . ceil () . long ()
         pos_floor = pos . floor () . long () 
         # query维度为[batch, seq_len, head_dim], pos_emb维度为[1, head_dim, npos_max]
         
         # logits_int维度为[batch, seq_len, npos_max]
         logits_int = torch . matmul ( query , self . pos_emb )
        
         logits_ceil = logits_int . gather ( -1 , pos_ceil )
         logits_floor = logits_int . gather ( -1 , pos_floor )
         w = pos - pos_floor
         return logits_ceil * w + logits_floor * (1 - w )  
         
         

class SelfAttn( nn . Module) :
     def __init__(self, npos_max , head_dim ) :
         super() . __init__ ()
         self . cope = CoPE ( npos_max , head_dim )
         self . head_dim = head_dim

     def forward(self, query , key , val , mask ) :
         # q, k, v have dimensions batch x seq_len x head_dim
         # attn_logits的维度为[batch_size, seq_len, seq_len]
         attn_logits = torch . bmm ( query , key . transpose ( -1 , -2) )
         attn_logits = attn_logits / math . sqrt ( self . head_dim )
         attn_logits += mask . log ()   
         
         # 对于decoder来讲, attn_logits与mask相加之后，只有下半部分起作用，右上半部分为负无穷。负无穷输入sigmoid为0
         attn_logits += self . cope ( query , attn_logits )
         attn = torch . softmax ( attn_logits , dim = -1)
         out = torch . bmm ( attn , val )
         return out
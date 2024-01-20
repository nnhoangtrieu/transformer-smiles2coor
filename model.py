import torch 
import torch.nn as nn 
from utils import clones 
import math 

class SourceAttention(nn.Module) :
    def __init__(self, dim_model, num_head) :
        super(SourceAttention, self).__init__()
        assert dim_model % num_head == 0, 'dim_model % num_head != 0'
        self.dim_model = dim_model
        self.num_head = num_head
        self.dim_head = dim_model // num_head

        self.Q = nn.Linear(dim_model, dim_model)
        self.K = nn.Linear(dim_model, dim_model)
        self.V = nn.Linear(dim_model, dim_model)

        self.out = nn.Linear(dim_model, dim_model)

    def forward(self, Q, K, V) :
        B = Q.size(0) 

        Q, K, V = self.Q(Q), self.K(K), self.V(V)

        len_Q, len_K, len_V = Q.size(1), K.size(1), V.size(1)

        Q = Q.reshape(B, self.num_head, len_Q, self.dim_head)
        K = K.reshape(B, self.num_head, len_K, self.dim_head)
        V = V.reshape(B, self.num_head, len_V, self.dim_head)
        
        K_T = K.transpose(2,3).contiguous()

        attn_score = Q @ K_T

        attn_score = attn_score / (self.dim_head ** 1/2)

        attn_distribution = torch.softmax(attn_score, dim = -1)

        attn = attn_distribution @ V

        attn = attn.reshape(B, len_Q, self.num_head * self.dim_head)
        
        attn = self.out(attn)

        return attn, attn_distribution
        
class TargetAttention(nn.Module) :
    def __init__(self, dim_model, num_head, longest_coor) : 
        super(TargetAttention, self).__init__() 
        
        self.dim_model = dim_model
        self.num_head = num_head
        self.dim_head = dim_model // num_head  
    
        self.Q = nn.Linear(dim_model, dim_model)   
        self.K = nn.Linear(dim_model, dim_model)
        self.V = nn.Linear(dim_model, dim_model)

        self.out = nn.Linear(dim_model, dim_model)
    
    def forward(self, Q, K, V, mask = None) :
        B = Q.size(0) 

        Q, K, V = self.Q(Q), self.K(K), self.V(V)

        len_Q, len_K, len_V = Q.size(1), K.size(1), V.size(1)

        Q = Q.reshape(B, self.num_head, len_Q, self.dim_head)
        K = K.reshape(B, self.num_head, len_K, self.dim_head)
        V = V.reshape(B, self.num_head, len_V, self.dim_head)

        
        K_T = K.transpose(2,3).contiguous()

        attn_score = Q @ K_T

        attn_score = attn_score / (self.dim_head ** 1/2)

        if mask is not None :
            attn_score = attn_score.masked_fill(mask == 0, -1e9)
        
        attn_distribution = torch.softmax(attn_score, dim = -1)

        attn = attn_distribution @ V

        attn = attn.reshape(B, len_Q, self.num_head * self.dim_head)
        
        attn = self.out(attn)

        return attn, attn_distribution
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    

class Encoder(nn.Module) :
    def __init__(self, dim_model, num_head, num_layer, dropout, vocab_size) :
        super(Encoder, self).__init__()
        self.dim_model = dim_model
        self.embed = nn.Embedding(vocab_size, dim_model)
        self.pe = PositionalEncoding(dim_model, dropout)
        self.layers = clones(EncoderLayer(dim_model, num_head, dropout), num_layer)       
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, x) :
        x = self.embed(x) * (self.dim_model ** 0.5) 
        x = self.pe(x)
        for layer in self.layers : 
            x = layer(x) 
        return self.norm(x) 


class EncoderLayer(nn.Module) :
    def __init__(self, dim_model, num_head, dropout)  :
        super(EncoderLayer, self).__init__()  

        self.norm1 = nn.LayerNorm(dim_model)
        self.drop1 = nn.Dropout(dropout)
        self.self_attn = SourceAttention(dim_model, num_head) 

        self.norm2 = nn.LayerNorm(dim_model)
        self.feed_foward = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(dim_model, dim_model)
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x) : 
        x = self.norm1(x)
        attn, self_attn = self.self_attn(x, x, x)
        x = x + self.drop1(attn)

        x = self.norm2(x)
        x = self.feed_foward(x)
        x = x + self.drop2(x)   

        return x
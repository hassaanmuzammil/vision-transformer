# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import *
from einops import rearrange, repeat

class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.Q = nn.Linear(embedding_dim, embedding_dim//2)
        self.K = nn.Linear(embedding_dim, embedding_dim//2)
        self.V = nn.Linear(embedding_dim, embedding_dim//2)
        
    def forward(self, x):
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        # transpose K for dot product with Q
        attn_scores = torch.bmm(Q, K.transpose(1,2)) / math.sqrt(self.embedding_dim)
        # softmax across embedding (column) dim
        attn_scores  = attn_scores.softmax(dim = 2)
        attn_weights = torch.bmm(attn_scores, V)

        return attn_weights

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_linear = nn.Linear(patch_size**2 * num_channels, pre_linear_dim)
        self.position_embedding = nn.Embedding(num_patches + 1, position_embedding_dim)
        self.class_embedding  = repeat(nn.Parameter(torch.randn(1, class_embedding_dim).to(device)), '() e -> b e', b=batch_size)
        
        self.multi_self_attention = []
        for i in range(num_attention_heads):
            self.multi_self_attention.append(SelfAttention(1024).to(device))
        
        self.batchnorm1 = nn.BatchNorm1d(num_patches + 1)
        self.multi_layer_perceptron = MultiLayerPerceptron()
        self.batchnorm2 = nn.BatchNorm1d(num_patches + 1)
        
        self.post_linear = nn.Linear(256, num_classes)
        
    def forward(self, x):
        
        if True:
            x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)# (b, num_patches, patch_size**2)
        
        x = F.relu(self.pre_linear(x)) # (32, 16, 512)
        pos = self.position_embedding(torch.tensor(range(num_patches+1)).to(device).repeat(batch_size, 1)) # (32, 17, 512)
        cl = self.class_embedding # (32, 512)

        cl_pos = torch.cat((cl, pos[:,0,:]), dim=-1) # (32, 1024)
        x_pos = torch.cat((x, pos[:,1:,:]), dim=-1) # (32, 16, 1024)

        x = torch.cat((cl_pos.unsqueeze(dim=1), x_pos), dim=1) # (32, 17, 1024)

        skip1 = x.clone().detach()
        x = self.batchnorm1(x)

        attn_outputs = []
        for self_attention in self.multi_self_attention:
            attn_outputs.append(self_attention(x))

        attn_outputs = torch.cat(attn_outputs, dim = -1)
        x = torch.cat((skip1, attn_outputs), dim=-1) # (32, 17, 2048)
        skip2 = x.clone().detach() # (32, 17, 2048)

        x = self.batchnorm2(x)
        x = self.multi_layer_perceptron(x) # (32, 17, 256)
        x = self.post_linear(x[:,0,:])  # (32, 10)   

        return x
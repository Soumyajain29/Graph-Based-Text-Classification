import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

class GCN(nn.Module):
  def __init__(self , config):
    super().__init__()
    self.vocab_size = config['vocab_size']
    self.layer1_dim = config['gcn_layer1_dim']
    self.layer2_dim = config['gcn_layer2_dim']
    self.layer1 = nn.Linear(self.vocab_size , self.layer1_dim)   #W1
    self.layer2 = nn.Linear(self.layer1_dim , self.layer2_dim)   #W2
    self.X      = torch.eye(self.vocab_size,device = config['device'])
    self.dropout = nn.Dropout(config['gcn_drop_1'])
  def forward(self , A):
    #print(self.X)
    #   GNN_out = Relu(AHW)  H = Relu(AXW)
    AX    = torch.matmul(A , self.X)
    H1    = self.layer1(AX)
    H1    = self.dropout(F.relu(H1))
    AH1   = torch.matmul(A , H1)
    H2   =  self.layer2(AH1)
    H2     = F.relu(H2)
    #print(H2.shape)
    return H1
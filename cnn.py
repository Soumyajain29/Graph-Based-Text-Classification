import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from gcn import GCN



class CNN(nn.Module):
  def __init__(self, config , pretrained_vecs , graph_embeddings):
    super().__init__()
    self.device                  =config['device']
    self.vocab_size              = config['vocab_size']
    self.gcn_embeddings          = nn.Embedding.from_pretrained(graph_embeddings , padding_idx = 0, freeze = False)
    self.pretrained_embeddings   = nn.Embedding.from_pretrained(pretrained_vecs , padding_idx = 0, freeze = False)
    self.in_channel              = 1          
    self.out_channel             = config['num_filters']   #of each size 
    self.cnn_input_dim           = config['embedding_dim'] + config['gcn_layer1_dim']
    self.filter_sizes            = config['filter_sizes']
    self.convolution_layers      = nn.ModuleList()
    self.output_dim              = config['num_classes']
    self.dropout1                = nn.Dropout(config['embed_drop'])
    for size in self.filter_sizes :
      self.convolution_layers.append(nn.Conv2d(self.in_channel, self.out_channel, (size, self.cnn_input_dim)))
    self.linear                  = nn.Linear(len(self.filter_sizes)*self.out_channel, self.output_dim)
    self.dropout                 = nn.Dropout(config['cnn_drop'])

  def forward(self ,X):
      gcn_x = self.gcn_embeddings(X)
      pretrained_embeddings = self.pretrained_embeddings(X)
      # X = gcn_x
      X = torch.cat([gcn_x , pretrained_embeddings] , dim = -1)
      X = self.dropout1(X)
      X = X.unsqueeze(1)     # x = [batch_size , ci , seq_len , input_him]
      conv_out = []
      for layer in self.convolution_layers: 
        layer_out = layer(X).squeeze(3)
        layer_out = F.relu(layer_out)
        conv_out.append(layer_out)      #before_squeeze = [ batch_size , co , seq_len-kernel_size+1]
      #print(x.shape) 
      h = [F.max_pool1d(x , x.size(2)).squeeze(2) for x in conv_out]  #before_squeeze = [batch_size , co ,1]
      #print(x.shape)
      h = torch.cat(h, 1)
      h = self.dropout(h)
      h = self.linear(h)
      #print(x.shape)
      return h
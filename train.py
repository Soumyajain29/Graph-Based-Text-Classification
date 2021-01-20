from config import config
import numpy as np
from evaluate import Eval
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from gcn import GCN
from cnn import CNN
from utils import save_checkpoint ,preprocess_adj
from copy import deepcopy
from data import Get_data
import pickle

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
CUDA_LAUNCH_BLOCKING=1

class train():
  def __init__(self , config):
    self.config = config
    self.evaluation = Eval(config)
    self.num_epoch = config['num_epoch']
    self.device  = config['device']
    self.embedding_dim = config['embedding_dim']
    self.vocab_dir = config['vocab_dir']
    self.embedding_dir = config['embedding_dir']
    self.adj_path  = config['adj_path']
    self.pretrained_embeddings = self.get_pretrained_emebeddings()
    self.adj_matrix = self.get_adjacency_matrix()

    self.data  = Get_data(self.config)
    self.train_iterator , self.val_iterator = self.data.read_data()
      
  def save(self,model):
    model_dir = self.config['model_dir']
    save_checkpoint(model, model_dir)

  def get_pretrained_emebeddings(self):
    glove_dict = {}
    with open(self.embedding_dir , 'r') as f:
      for line in f:
        line = line.split()
        #print(line)
        glove_dict[line[0]] = [float(x) for x in line[1:]]

    c=0
    pad_vec = list(np.zeros(self.embedding_dim))
    unk_init = list(np.random.uniform(low = -.25 , high = .25 , size = self.embedding_dim))
    vocab_vec = []
    vocab_ind = {}
    index = 0
    with open(self.vocab_dir,'r') as f:
      for line in f:
        line = line.strip()
        vocab_ind[line] = index
        index += 1
        if line in glove_dict:
          c += 1 
          vocab_vec.append(glove_dict[line])
        else:
          vocab_vec.append(unk_init)

    vocab_vec[0] = pad_vec
    print("words of my vocab that are meaningful in glove is ", c)
    vocab_vec = torch.Tensor(vocab_vec).to(self.device)
    return vocab_vec
  
  def get_adjacency_matrix(self):
    with open(self.adj_path , 'rb') as f :
      adj_matrix = pickle.load(f)
    adj_matrix = preprocess_adj(adj_matrix)
    print('shape of adjaceny_matrix' , adj_matrix.shape)
    adj_matrix = torch.Tensor(adj_matrix)
    return adj_matrix

  def training(self):
    best_epoch_loss = float('inf')
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []
    best_adf = None

    gcn_model                    = GCN(config).to(self.device)
    graph_embeddings             = gcn_model(self.adj_matrix.to(self.device))
    pad_vec                      = torch.zeros((1, config['gcn_layer1_dim']) , device = self.device)
    unk_init                     = torch.FloatTensor(1, config['gcn_layer1_dim'] ).uniform_(-.25, .25)
    unk_init                     = unk_init.to(self.device)
    graph_embeddings             = torch.cat([pad_vec , unk_init , graph_embeddings])
    model = CNN(self.config , self.pretrained_embeddings , graph_embeddings)
    model = model.to(self.device)
    optimizer = optim.Adam(list(gcn_model.parameters())+list(model.parameters()) 
               , lr = self.config['learning_rate'] , weight_decay = self.config['l2_regularization'])
    criterian = nn.CrossEntropyLoss(reduction = 'sum')
    model.train()
     
    for epoch in range(self.num_epoch):
      epoch_loss = 0.0
      num_int = 0
      epoch_acc = 0.0
      for i , batch in enumerate(self.train_iterator):
        text , _  = batch[0]
        text = text.to(self.device)
        labels  = batch[1].to(self.device)
        num_int+= len(labels)
        #print(batch_labels)
        optimizer.zero_grad()
        pred   = model(text)
        #print(pred)
        loss = criterian(pred , labels)
        num_correct  = (torch.max(pred,1)[1].view(labels.size()).data == labels.data).float().sum()
        epoch_acc += num_correct
        # reg_loss = None
        # for param in model.parameters():
        #   if reg_loss is None:
        #       reg_loss = 0.5 * torch.sum(param**2)
        #   else:
        #       reg_loss = reg_loss + 0.5 * param.norm(2)**2

        factor = self.config['l2_regularization']
        #loss += factor * reg_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()  
    
      val_loss , epoch_val_acc, analysis_df  = self.evaluation.evaluate(model , criterian , self.val_iterator)
      
      if val_loss < best_epoch_loss:
        best_epoch_loss = val_loss
        best_epoch = epoch
        model_copy = deepcopy(model)
        best_adf = analysis_df
      
      train_acc.append((epoch_acc/num_int) * 100)
      train_losses.append(epoch_loss/num_int)
      val_losses.append(val_loss)
      val_acc.append(epoch_val_acc)
     #print(analysis_df[:15])

      print('''epoch : {}, train_loss : {}, train_acc : {} , val_loss : {} , val_acc : {}'''.format(epoch ,  
          epoch_loss/num_int, epoch_acc/num_int , val_loss , epoch_val_acc))
    
    print(best_epoch_loss, best_epoch)
    self.save(model_copy)
    return train_losses , val_losses , train_acc , val_acc, model_copy , best_adf

if __name__ == "__main__" :
  obj = train(config)
  train_losses , val_losses , train_acc , val_acc ,  model_copy , best_adf = obj.training()
  print(val_acc)
  print(best_adf[15:30])
  epochs = [x for x in range(config['num_epoch'])]
  fig1 = plt.figure(1)
  plt.plot(epochs , train_losses , label = 'train_loss' , color = 'red')
  plt.plot(epochs , val_losses , label = 'val_loss' , color = 'blue')
  #plt.plot(epochs , test_losses , label = 'test_loss' , color = 'brown')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title("Epoch vs Loss")
  plt.legend()
  plt.savefig('2.png')
  fig2 = plt.figure(2)
  plt.plot(epochs , train_acc , label = 'train_acc' , color = 'red')
  plt.plot(epochs , val_acc , label = 'val_acc' , color = 'blue')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.title("Epoch vs Accuracy")
  plt.legend()
  plt.savefig('1.png')
  #plt.show()
  plt.close()
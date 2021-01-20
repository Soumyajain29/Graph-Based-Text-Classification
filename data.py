import torch
from config import config
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class Dataset(Dataset):
  def __init__(self, text, labels):
    self.text = text
    print( ' label type in dataset' ,type(labels) , len(labels))
    self.labels = torch.Tensor(labels)
 
  def __getitem__(self, idx):
    x = torch.tensor(self.text[idx] , dtype = torch.long)
    y = self.labels[idx]
    return (x,y)
 
  def __len__(self):
    return self.labels.shape[0]

class Iterator():
  def __init__(self,config):
    self.batch_size = config['batch_size']

  def collate_fn(self ,batch):
    lengths = [item[0].shape[0] for item in batch]
    max_length = max(lengths)
    data = []
    for (item , _ ) , length in zip(batch , lengths):
      data.append(torch.cat([item , torch.zeros(max_length - length)]))
    targets = [item[1] for item in batch]
    data = torch.stack(data)
    data = data.long()
    lengths = torch.Tensor(lengths)
    lengths = lengths.long()
    targets = torch.Tensor(targets)
    targets = targets.long()
    return [(data , lengths), targets]

  def get_iterator(self ,dataset):
    iterator = DataLoader(dataset , batch_size = self.batch_size , collate_fn= self.collate_fn)
    return iterator

class Get_data():
  def __init__(self,config):
    self.config = config
    self.data_path = self.config['data_path']
    self.iterator  = Iterator(self.config)
    self.vocab_dir = config['vocab_dir']
    self.vocab     = self.build_vocab()

  def build_vocab(self):
    vocab = {}
    index = 0
    with open(self.vocab_dir,'r') as f:
      for line in f:
        line = line.strip()
        vocab[line] = index
        index += 1
    return vocab

  def tokenization(self,text):
    indices = []
    for word in text.split():
      if word in self.vocab:
        indices.append(self.vocab[word])
      else:
        indices.append(self.vocab['unk_t'])
    return indices

  def read_data_(self,ds):
    if ds == 'r8':
      label_index = self.config['r8_labels']#{'acq':0, 'trade':1, 'crude':2, 'ship':3, 'interest':4, 'money-fx':5, 'grain':6, 'earn':7}
    if ds == '20ng':
      label_index = {'rec.autos':0,'comp.sys.mac.hardware':1, 'rec.sport.hockey':2, 'talk.politics.guns':3, 'talk.politics.mideast':4, 
    'sci.space':5, 'soc.religion.christian':6, 'rec.motorcycles':7, 'sci.crypt':8, 'comp.graphics':9, 'talk.politics.misc':10, 
    'sci.med':11, 'talk.religion.misc':12, 'sci.electronics':13, 'comp.os.ms-windows.misc':14, 'alt.atheism':15, 
    'comp.sys.ibm.pc.hardware':16, 'misc.forsale':17, 'rec.sport.baseball':18,'comp.windows.x':19 }

    text = []
    label = []
    with open(self.config['data_folder']+'/'+ds+'.clean.txt','r') as f:
      for line in f:
        line = line.strip()
        text.append(line)
    f.close()
    with open(self.config['data_folder']+'/'+ds+'.txt','r') as f:
      for line in f:
        line =  line.split('\t')[2]
        label.append(label_index[line.strip()])   
    return text, label

  def read_data(self):
    text = []
    labels=[]
    if self.config['dataset'] == 'mr':
      with open(self.data_path,'rb') as f:
        for line in f:
          line  = line.decode('latin-1')
          # print(line)
          labels.append(int(line[0]))
          text.append(line[2:].strip())
    else:
      text,labels = self.read_data_(self.config['dataset'])
    
    
    df = pd.DataFrame(text,columns=['text'])
    df['label'] = labels

    #tokenize 
    df['text'] = df['text'].apply(self.tokenization)

    #shuffle
    df=df.sample(frac=1).reset_index(drop=True)

    #split
    train_x , val_x , train_y , val_y = train_test_split(df['text'] ,  df['label'] , test_size = .2 , stratify = df['label'] )

    #get tensor_dataset
    train_set = Dataset(train_x.reset_index(drop = True) , train_y.reset_index(drop = True))
    print('i am here' , type(train_y) , type(val_y))
    print(val_y)
    val_set   = Dataset(val_x.reset_index(drop = True) , val_y.reset_index(drop = True))

    #get_iterator
    train_iterator = self.iterator.get_iterator(train_set)
    val_iterator   = self.iterator.get_iterator(val_set)

    return train_iterator , val_iterator

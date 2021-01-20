from config import config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pandas as pd

class Eval():
  def __init__(self , config):
    self.config = config
    self.device = config['device']
 
  def evaluate(self,model , criterian ,val_iterator) :
    model.eval()
    analysis_df = pd.DataFrame()
    epoch_acc = 0.0
    with torch.no_grad():
      val_loss = 0.0
      num_int = 0
      for i , batch in enumerate(val_iterator):
        text , _  = batch[0]
        text = text.to(self.device)
        labels  = batch[1].to(self.device)
        num_int+= len(labels)
        pred  = model(text)
        #print(pred.shape)
        num_correct  = (torch.max(pred,1)[1].view(labels.size()).data == labels.data).float().sum()
        epoch_acc += num_correct
        loss = criterian(pred , labels)
        val_loss += loss.item()

        # if i == 0 :
        #   analysis_df['true_labels'] = [0,0,0]
        #   analysis_df['prediction']  = pred

    return val_loss/num_int , epoch_acc/num_int , analysis_df

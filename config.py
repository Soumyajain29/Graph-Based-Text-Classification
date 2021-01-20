import torch

s_base = '/content/drive/MyDrive/DLNLP/project/'
d_base = '/content/drive/MyDrive/project/'
config = {
'embedding_dir'   :  s_base + 'data/glove.6B.300d.txt' ,
'vocab_dir'       :  s_base + 'data/mrvocab.txt',
'embed_drop'      :  0,
'cnn_drop'        :  0.8,
'gcn_drop_1'      :  0.8,
'vocab_size'      :  21401 , #7688: R8 and 21401:MR and 42757: 20ng
'batch_size'      :  32,
'gcn_layer1_dim'  :  128,
'gcn_layer2_dim'  :  64,
'num_classes'     :  2 ,
'num_epoch'       :  100,
'embedding_dim'   :  300, 
'learning_rate'   :  5e-5 ,
'l2_regularization' : 1e-4,
'filter_sizes'    :  [3,4],
'num_filters'     :  100 ,
'window'          :  40 , #NOT AFFECTING STILL 20 
'device'          :  'cuda' if torch.cuda.is_available() else 'cpu' ,
'dataset'         :  'mr' ,
'adj_path'        :  'data/mr.adj',
'data_path'       :  s_base + 'data/rt-polarity.all.txt',
'model_dir'       :  s_base + 'checkpoints/model.bin',
'r8_labels'       :  {'acq':0, 'trade':1, 'crude':2, 'ship':3, 'interest':4, 'money-fx':5, 'grain':6, 'earn':7}
}


#pad and unk added in vocab.txt
# Graph-based-text-classification

This is code for our proposed model in DL-NLP course.
![See proposed model here](https://github.com/deekshakoul/Graph-based-text-classification/blob/main/dlnlp.jpg)

* Steps to run this model:
1. Datasets of MR, SST-2, R8 and 20ng should be put under Data/ and path needs to be updated in config.py file
2. For parameter tuning, use config.py file and change parameters.
3. python3 train.py

For the baselines mentioned in results, please refer follwoing codes that we had implemeneted:

* TF-IDF with Logistic Regression:[TF-IDF + LR]( https://github.com/deekshakoul/Sentiment-Analysis-for-movie-reviews.git)
* LSTM with pre-trained GloVe embeddings(d=300) : [LSTM - GloVe](https://github.com/deekshakoul/Examples-of-DL-NLP-using-Pytorch.git)
* For BERT, we used a package [Simple Transformers](https://simpletransformers.ai/)
* Code for TGCN and VGCN-BERT: gcn.py + adjacency.ipynb + train.py

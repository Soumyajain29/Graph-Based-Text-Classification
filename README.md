# Graph-Based-Text-Classification

This code is for our proposed model in DL-NLP course.
![See proposed model here](https://github.com/Soumyajain29/Graph-Based-Text-Classification/blob/main/images/model.jpg)

* Steps to run this model:
1. Datasets of MR, SST-2, R8 and 20ng should be put under Data/ and path needs to be updated in config.py file.
2. For parameter tuning, use config.py file and change parameters.
3. python3 train.py

Baselines:

* TF-IDF with Logistic Regression:[TF-IDF + LR]( https://github.com/deekshakoul/Sentiment-Analysis-for-movie-reviews.git)
* LSTM with pre-trained GloVe embeddings(d=300) : [LSTM - GloVe](https://github.com/deekshakoul/Examples-of-DL-NLP-using-Pytorch.git)
* Code for TGCN and VGCN-BERT: adjacency.ipynb + gcn.py + train.py 

* Dataset Statistics:
![Dataset Statistics](https://github.com/Soumyajain29/Graph-Based-Text-Classification/blob/main/images/dataset_statistics.jpg)

* Results:
![results](https://github.com/Soumyajain29/Graph-Based-Text-Classification/blob/main/images/results.jpg)

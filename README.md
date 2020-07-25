# Text-Sentiment-Predictor
This application takes input as a text message and outputs an emoji and sentiment for that message.

# Dependencies
1. numpy
2. pandas
3. keras
4. tensorflow
5. emoji
6. nltk

# Working
For training this model takes the text message and convert that to word vectors using https://nlp.stanford.edu/projects/glove/glove.6B.zip word vectorization.
The summary of various layers of trained model is as follows:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 40)                0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 40, 300)           120000300 
_________________________________________________________________
bidirectional_1 (Bidirection (None, 40, 256)           439296    
_________________________________________________________________
dropout_1 (Dropout)          (None, 40, 256)           0         
_________________________________________________________________
lstm_2 (LSTM)                (None, 128)               197120    
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 7)                 903       
_________________________________________________________________
activation_1 (Activation)    (None, 7)                 0         
=================================================================
Total params: 120,637,619
Trainable params: 637,319
Non-trainable params: 120,000,300

#Download trained model
You can download the trained model from my personal google drive https://drive.google.com/file/d/11bAyM6irdjZxsYn7Kg6P8w_cbJtizA5M/view?usp=sharing
You can download the glove vectors from https://drive.google.com/file/d/1-B0joQ0-FGFd17k7FiDP0RCirgzC8bOF/view?usp=sharing
Save both these files in the same directory where all other files are, otherwise change the train.py and test.py codes so that they can access these files. 

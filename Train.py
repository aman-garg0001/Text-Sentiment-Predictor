import pandas as pd
import numpy as np
import emoji

data_set = pd.read_csv('emoji_data.csv', engine = 'python')

text = data_set['Text']
label = data_set['Label']

emoji_dict = {"joy":"😂", "fear":"😱", "anger":"😠", "sadness":"😢", "disgust":"😒", "shame":"😔", "guilt":"😳"}
emoji = {"joy":0, "fear":1, "anger":2, "sadness":3, "disgust":4, "shame":5, "guilt":6}

for i in range(len(label)):
  label[i] = emoji[label[i]]
 
Y = np.zeros((len(label), 7))
for i in range(len(label)):
  Y[i, label[i]] = 1
  
def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip

words_to_index, index_to_words, word_to_vec_map = read_glove_vecs('glove.6B.300d.txt')
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform

from nltk.tokenize import TweetTokenizer

def sentences_to_indices(X, words_to_index, max_len):
    m = X.shape[0]                                   
    X_indices = np.zeros((m, max_len))
    tknz = TweetTokenizer()
    for i in range(m):                             
        sentence_words = [word.lower() for word in tknz.tokenize(X[i])]
        j = 0
        for w in sentence_words:
            if j > 39:
              break
            try:
              X_indices[i, j] = words_to_index[w]
              j = j + 1
            except KeyError:
              continue  
    return X_indices
    
X_set = np.array(text)
max_len = 40
X_set = sentences_to_indices(X_set, words_to_index, max_len)

vocab_len = len(words_to_index) + 1   
emb_dim = word_to_vec_map["cucumber"].shape[0]      
emb_matrix = np.zeros((vocab_len, emb_dim))
for word, idx in words_to_index.items():
    emb_matrix[idx, :] = word_to_vec_map[word]
embedding_layer = Embedding(vocab_len, emb_dim, trainable = False)
embedding_layer.build((None,))
embedding_layer.set_weights([emb_matrix])

sentence_indices = Input((max_len, ), dtype = 'int32')
    
embeddings = embedding_layer(sentence_indices)

X = Bidirectional(LSTM(units = 128, return_sequences = True))(embeddings)

X = Dropout(rate = 0.4)(X)

X = LSTM(units = 128, return_sequences = False)(X)

X = Dropout(rate = .4)(X)

X = Dense(units = 7)(X)

X = Activation('softmax')(X)

model = Model(inputs = sentence_indices, outputs = X)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_set, Y, test_size=0.10)

model.fit(X_train, y_train, epochs = 20, batch_size = 32, shuffle=True, validation_data=(X_test, y_test))

model.save('Emoji-Predictor.h5')

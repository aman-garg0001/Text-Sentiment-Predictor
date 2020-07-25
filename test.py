import pandas as pd
import numpy as np
import emoji
from nltk.tokenize import TweetTokenizer
import keras 
import tensorflow

emoji_dict = {"joy":"😂", "fear":"😱", "anger":"😠", "sadness":"😢", "disgust":"😒", "shame":"😔", "guilt":"😳"}
emoji = {"joy":0, "fear":1, "anger":2, "sadness":3, "disgust":4, "shame":5, "guilt":6}

def preprocess(X, words_to_index, max_len):                                   
    X_indices = np.zeros((1, max_len))
    tknz = TweetTokenizer()
    sentence_words = [word.lower() for word in tknz.tokenize(X)]
    j = 0
    for w in sentence_words:
            if j > 39:
              break
            try:
              X_indices[0, j] = words_to_index[w]
              j = j + 1
            except KeyError:
              continue  
    return X_indices
   
max_len = 40
words_to_index, index_to_words, word_to_vec_map = read_glove_vecs('glove.6B.300d.txt')

file = open('input.txt', 'r')
inp = file.read()
file.close()
x = preprocess(inp, words_to_index, max_len)
model = keras.models.load_model('Emoji-Predictor.h5')

y = model.predict(x)
y = np.argmax(y, axis = -1)
arr = list(emoji_dict.keys())

ans = "Predicted Sentiment: " + arr[y[0]]
ans += "\nPredicted Emoji: " + emoji_dict[arr[y[0]]]

file = open('output.txt', 'w')
file.write(ans)
file.close()

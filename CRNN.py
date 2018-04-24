import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import shutil
import os
import os.path
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from os.path import join, dirname
import random
import gensim
import gensim.models.keyedvectors as word2vec
import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Embedding,Concatenate,Bidirectional,LSTM,Dropout
from keras.models import Model

from keras.models import Sequential
import keras.preprocessing.sequence as S


random.seed(0)
_COL_NAMES = ['line_number', 'speaker', 'text', 'label']
_COL_DICT = ['word']


MAX_SEQUENCE_LENGTH = 2000 # 每句话选取50个词
MAX_NB_WORDS = 12000 # 将字典设置为含有1万个词
EMBEDDING_DIM = 300 # 词向量维度，300维
VALIDATION_SPLIT = 0.25 # 测试集大小，全部数据的20%


def get_vect4sent(sent,model):
    # split sent into word_list
    word_list = sent.split(" ")
    # get vect for each of word
    word_vect = []
    for word in word_list:
        if word in model.vocab:
            word_vect.append(model[word])
    # get average vector for the sent
    avg_vect = np.mean(word_vect,axis=0)
    return avg_vect

def load_wordList():
    path = '/home/dongsheng/code/CLEF/clef2018-factchecking-master/CNN/dict/wordList2.txt'
    with open(path, 'rb') as f:
        lines = [l.decode('utf8', 'ignore').strip() for l in f.readlines()]
    #words = pd.read_csv(path, names=_COL_DICT, sep='\t')
    return lines

def load_data():
    data = []
    count=0
    for root, dirs, files in os.walk("/home/dongsheng/code/CLEF/clef2018-factchecking-master/data/res2"):
        for name in files:
            if name.endswith("txt"):
                with open(root + '/' + name,'r') as f:
                    countLine=0
                    sent2year=[]    # first value
                    ruling=-1   # second value
                    
                    sample=[]
                    for line in f:
                        countLine=countLine+1
                        if(countLine==1):
                            ruling=line.strip()
                        else:
                            strs = line.strip().split("\t")
                            if len(strs)<3:
                                continue
                            sents.append([strs[3])
                            totreturn=strs[0]
                    sample.append(".".join(sents))
                    if ruling in ['False','Pants on Fire!','Mostly False']:
                        ruling=0
                    elif ruling in ['Mostly True','','True']:
                        ruling=2
                    elif ruling=='Half-True':
                        ruling=1
                    else:
                        print(ruling)
                        ruling=-1      
                    sample.append(ruling)      
                    data.append(sample)
                    #print(count,len(sample[0]),sample[1])
                    count=count+1
    print("data size:",count)               
    return np.array(data)
    
def get_vect2label(df,model):
    vect_list=[]
    label_list=[]
    for sent in df['text']:
        avg_vect=get_vect4sent(sent,model)
        vect_list.append(avg_vect)
    for label in df['label']:
        if label=='1':
            label_list.append([0,1])
        else:
            label_list.append([1,0])
    return vect_list,label_list

if __name__=='__main__':
    model = word2vec.KeyedVectors.load_word2vec_format('/home/dongsheng/code/CLEF/clef2018-factchecking-master/CNN/dict/GoogleNews-vectors-negative300.bin', binary=True)
    #S1: embedding index: 1w*300 -> space
    embeddings_index = {}
    wordList = load_wordList()
    print(len(wordList))
    word_vectors = model.wv
    for word in wordList:
        if word in model.vocab:
            if len(embeddings_index)<MAX_NB_WORDS:
                #print(word)
                embeddings_index[word] = word_vectors[word]
    #word_vectors = model.wv
    #for word, vocab_obj in model.wv.vocab.items():
    #    if int(vocab_obj.index) < MAX_NB_WORDS:
    #        embeddings_index[word] = word_vectors[word]
    #del model, word_vectors # 删掉gensim模型释放内存
    print('Found %s word vectors.' % len(embeddings_index))

    #S2: read data: text = training_X, labels = training_Y, labels_index = HashMap<String,String>
    texts = []  # list of text samples
    labels = []  # list of label ids
    labels_index = {}  # label与name的对应关系
    # manually 
    labels_index["0"] = 0
    labels_index["1"] = 1
    labels_index["2"] = 2
    
    all_data = load_data()
    texts = all_data[:,0]
    labels = all_data[:,1]

    
    print('Found %s texts.' % len(texts))
    print('Found %s label.' % len(labels_index))
    
    #S3: text preprocessing:  
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS) # 传入我们词向量的字典
    tokenizer.fit_on_texts(texts) # 传入我们的训练数据，得到训练数据中出现的词的字典
    sequences = tokenizer.texts_to_sequences(texts) # 根据训练数据中出现的词的字典，将训练数据转换为sequences

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) # 限制每篇文章的长度

    labels = to_categorical(labels) # label one hot表示
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    
    #S4: split training and validation dataset
    # shuttle
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    # Split data: 8:2
    x_train = data[:-num_validation_samples] 
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    #S5: embedding layer
    num_words = min(MAX_NB_WORDS, len(word_index))  # 对比词向量字典中包含词的个数与文本数据所有词的个数，取小
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # 文本数据中的词在词向量字典中没有，向量为取0；如果有则取词向量中该词的向量
            embedding_matrix[i] = embedding_vector
            
    # build models
    rnn_model = Sequential()
    rnn_model.add(Embedding(num_words, EMBEDDING_DIM,weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,trainable=False))
    rnn_model.add(Bidirectional(LSTM(128,implementation=2)))
    rnn_model.add(Dropout(0.5))
    rnn_model.add(Dense(len(labels_index), activation='softmax'))
    rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])  # optimizer=RMSprop
    
    # 如果希望短一些时间可以，epochs调小
    rnn_model.fit(x_train, y_train,
          batch_size=128,
          epochs=20)
    res = rnn_model.evaluate(x_val,y_val)
    print(res)
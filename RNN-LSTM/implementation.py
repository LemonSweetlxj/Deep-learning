import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile
import gzip
import string
import sys
import collections



"""
This project is finished by Liyuan Song and Xinjie Lu on 14/10/2017.

There are main 5 steps in this whole project.
    Step 1: load the word dictionary
    Step 2: load the training data
    Step 3: build the model
    Step 4: training model
    Step 5: testing model
    
The problem we counter:
    1. The index of dictionary should start at 1 instead of 0.
    The index of 0 in dictionary should be UNK words and the vector of UNK should
    be replace by [0] *50
    
    2. The training accuracy remains 0.5 and does not change.
    It is because the predictions we got are wrong, we only get one demension vector but the right
    predictions should be 2-D that each demension indicates the probability of pos/neg
    So we change the fully-connection layer to weight and bias, which seems work.
    
    3.The parameter setting.
    In order to avoid underfitting and overfitting, it took us a long time to set parameter.
    learning rate: 1, 0,1, 0.01, 0.001
    lstm_size: 12,36,46,64, 128, 256
    dropout_keep_prob: 0.3, 0.5, 0.7, 1.0
    lstm_layer:1, 2
    First, we use learningrate:0.01, lstm_size:256, dropout_keep_prob:1.0, layer:2.
    At begining(the first 20000 iterations), the accuracy up to 98%, then it decreased dramaticly to around 50%--55%,
    and the testing accuracy only achieve 53%, which means our model underfitting.
    The secend set of parameter we set is learning_rate:0.001,lstm_size,dropout_keep_prob 1.0,layer:2.
    The accuracy of training up to 100%(in the first 10000 interations) and remains same, the test accuracy is low,
    this situation means that our model overfitting.
    To solve this problem ,we setting the parameter:
    learning rate:0.001
    lstm_size:12
    dropout_keep_prob:0.5
    lstm_layer:2
    
The result:
    The check step we use is 1000 instead of 50.
    We split the 25000 data by hand.
    After 10000 iterations, the training accuracy is around 80-85%
    and the average testing accuracy get 70%
"""
batch_size = 50

def check_file(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        print("please make sure {0} exists in the current directory".format(filename))
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            "File {0} didn't have the expected size. Please ensure you have downloaded the assignment files correctly".format(filename))
    return filename

def extract_data(filename):
    """Extract data from tarball and store as list of strings"""
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data2/')):
        with tarfile.open(filename, "r") as tarball:
            dir = os.path.dirname(__file__)
            tarball.extractall(os.path.join(dir, 'data2/'))
    return

def load_data(glove_dict):## training data
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""
    
    filename =check_file('reviews.tar.gz',14839260)
    extract_data(filename) # unzip

    ##read data
    print("READING DATA")
    data = []
    all_data = []  ##all_data in reviews
    reviews = []  ##all_review
    dir = os.path.dirname(__file__)
    file_list = glob.glob(os.path.join(dir,
                                        'data2/pos/*'))
    file_list.extend(glob.glob(os.path.join(dir,
                                        'data2/neg/*')))
    print("Parsing %s files" % len(file_list))
    for f in file_list:
        with open(f, encoding = 'utf8') as openf:
            s = openf.read()
            no_punct = ''.join(c for c in s if c not in string.punctuation)
            data.extend(no_punct.split())
            review = no_punct.split()
            temp = []
            ##change the word in review to lower case
            for i in review:
                temp.append(i.lower())
            reviews.append(temp)

    ##all_words with lower case
    for word in data:
        all_data.append(word.lower())

    ##change the word to number
    reviews_ints = []
    for review in reviews:
        temp = []
        for word in review:
            if word in glove_dict:
                temp.append(glove_dict[word])
            else:
                word = 'UNK'
                temp.append(glove_dict[word])
        reviews_ints.append(temp)

    # Filter out that review with 0 length
    ##For reviews shorter than 40, we'll pad with 0s.
    ##For reviews longer than 40, we can truncate them to the first 40 characters.
    reviews_ints = [r[0:40] for r in reviews_ints if len(r) > 0]

    seq_len = 40
    data = np.zeros((len(reviews_ints), seq_len), dtype=int)
    for i, row in enumerate(reviews_ints):
        data[i, -len(row):] = np.array(row)[:seq_len]

    return data


def load_glove_embeddings():##dictionary
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    global n_words
    data = open("glove.6B.50d.txt",'r',encoding="utf-8")
    vocab = ['UNK']  #if there exist some words in review that not in dictionary
    embed = []       #assign it to UNK.
    embed_unk = [0] *50  #the vector of UNK is 50*0s
    embed.append(embed_unk)
    word_index_dict = {}
    for line in data.readlines():
        row = line.strip().split(' ')
        word = row[0]
        vocab.append(word)
        embed.append([float(val) for val in row[1:]])
    embeddings = np.array(embed, dtype = np.float32)
    for word in vocab:
        word_index_dict[word] = vocab.index(word)

    #print("word_index_dict")
    n_words = len(word_index_dict)
    data.close()
    
    #if you are running on the CSE machines, you can load the glove data from here
    #data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
    return embeddings, word_index_dict


def lstm_cell():
    #use multiple LSTM cells with dropout
    lstm_size = 12
    #keep_prob the probability of keeping weights in the dropout layer
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=dropout_keep_prob)

def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, optimizer, accuracy and loss
    tensors"""
    #lstm_size is number of units in the hidden layers in the LSTM cells. Common values are 128, 256, etc
    #lstem_layers is number of LSTM layers in the network.
    #batch_size is the number of reviews to feed the network in one training pass
    lstm_size = 12
    lstm_layers = 2
    learning_rate = 0.001
    
    input_data = tf.placeholder(tf.int32, [batch_size, 40], name = "input_data")
    labels = tf.placeholder(tf.int32, [batch_size, 2], name = "labels")
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())

    #Embedded vectors pass to the LSTM cell.
    embedding = tf.nn.embedding_lookup(glove_embeddings_arr, input_data)
    embedding = tf.cast(embedding, tf.float32)
    embedding = tf.unstack(embedding, axis = 1)
    cell = lstm_cell()
    initial_state = cell.zero_state(batch_size, tf.float32)
    
    #run the data through the RNN nodes
    # The value of state is updated after processing each batch of words.
    outputs, final_state = tf.nn.static_rnn(cell, embedding, initial_state = initial_state, dtype = tf.float32)
    weight = tf.Variable(tf.random_normal(stddev = 1.0, shape = [lstm_size, 2]))
    bias = tf.Variable(tf.constant(value = 0.1, shape = [2]))
    # The LSTM output can be used to make next word predictions
    logits = tf.matmul(outputs[-1], weight) +bias
    predictions = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels),name = "loss")
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss = loss)
    ##checking Validation accuracy
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name = "accuracy")
        
    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss


"""
You are encouraged to edit this file during development, however your final
model must be trained using the original version of this file. This file
trains the model defined in implementation.py, performs tensorboard logging,
and saves the model to disk every 10000 iterations. It also prints loss
values to stdout every 50 iterations.
"""


import numpy as np
import tensorflow as tf
from random import randint
import datetime
import os

import implementation as imp

batch_size = imp.batch_size
iterations = 61000
seq_length = 40  # Maximum length of sentence

checkpoints_dir = "./checkpoints"

def getTrainBatch():
    labels = []
    arr = np.zeros([batch_size, seq_length])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(0, 11499)
            labels.append([1, 0])
        else:
            num = randint(13500, 24999)
            labels.append([0, 1])
        arr[i] = training_data[num]
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batch_size, seq_length])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(11500, 12499)
            labels.append([1, 0])
        else:
            num = randint(12500, 13500)
            labels.append([0, 1])
        arr[i] = training_data[num]
    return arr, labels

# Call implementation
glove_array, glove_dict = imp.load_glove_embeddings()
training_data = imp.load_data(glove_dict)
input_data, labels, dropout_keep_prob, optimizer, accuracy, loss = \
    imp.define_graph(glove_array)

# tensorboard
train_accuracy_op = tf.summary.scalar("training_accuracy", accuracy)
tf.summary.scalar("loss", loss)
summary_op = tf.summary.merge_all()

# saver
all_saver = tf.train.Saver()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

logdir = "tensorboard/" + datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

acc_train = 0
for i in range(iterations):
    batch_data, batch_labels = getTrainBatch()
    sess.run(optimizer, {input_data: batch_data, labels: batch_labels, dropout_keep_prob: 0.5})
    if (i % 50 == 0):
        loss_value, accuracy_value, summary = sess.run(
            [loss, accuracy, summary_op],
            {input_data: batch_data,
             labels: batch_labels, dropout_keep_prob: 0.5})
        acc_train += accuracy_value
        
        writer.add_summary(summary, i)
        print("Iteration: ", i)
        print("loss", loss_value)
        print("acc", accuracy_value)
    if (i % 10000 == 0 and i != 0):
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        save_path = all_saver.save(sess, checkpoints_dir +
                                   "/trained_model.ckpt",
                                   global_step=i)
        print("Saved model to %s" % save_path)


acc = 0
for i in range(1500):
    batch_data_1, batch_labels_1 = getTestBatch()
    test_loss, test_accuracy = sess.run(
            [loss, accuracy],
            {input_data: batch_data_1,
             labels: batch_labels_1, dropout_keep_prob: 1.0})
    acc += test_accuracy
    if (i % 10 == 0):
        print("test Iteration: ", i)
        print("test loss", test_loss)
        print("test acc", test_accuracy)
print("average acc:", acc/1500)       
sess.close()


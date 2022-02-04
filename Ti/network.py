#System print##################################
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from time import time

#Libraries####################################
import numpy as np
import random
from model_synapse import ALIF
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy.random as rd
import sys

#Parameters####################################
seed=1
train_size = 1594
test_size = 2542
time_step = 700
n_classes = 10
n_inputs = 78
n_hidden = 100
n_outputs = n_classes

n_batch = 100
learning_rate = 0.1
epoch = 100

t_ref=1
tau_m=5
tau_w=1
Vth=0.9
beta=10
tau_syn=50
Vmax=1.5 
tau_post=15 

print("seed:", seed, "train_size:", train_size, "test_size", test_size, \
"time_step:", time_step, "network:", n_inputs, n_hidden, n_classes,\
"n_batch:", n_batch, "lr:", learning_rate)
print("ref:",t_ref, "tau_m:", tau_m, "tau_w:", tau_w, \
"Vth:", Vth, "beta:", beta, "tau_syn:", tau_syn, "Vmax:", Vmax, "tau_p", tau_post)

#Read in Ti_digit dataset######################
trainData, testData=[], []
for i in range (n_classes):
		f=open("./Ti_digit/train/"+str(i))
		lines=f.readlines()
		for line in lines:
				trainData.append([i, list(map(int,line.split()))])

for i in range (n_classes):
		f=open("./Ti_digit/test/"+str(i))
		lines=f.readlines()
		for line in lines:
				testData.append([i, list(map(int,line.split()))])

random.seed(1)
random.shuffle(trainData)
random.shuffle(testData)

#Build dataset###################################
train_data, train_labels=[],[]
for i in range(train_size):
		label, data=trainData[i][0], trainData[i][1]
		preData = np.zeros((time_step, n_inputs))
		for j in range(len(data)):
				if j>0 and j%2==0:
						preData[data[j-1]][data[j-2]]=1
		
		train_data.append(preData)
		train_labels.append(label)

train_data=np.reshape(train_data, [train_size//n_batch, n_batch, time_step, n_inputs])
train_labels=np.reshape(train_labels, [train_size//n_batch, n_batch, ])

test_data, test_labels=[],[]
for i in range(test_size):
		label, data=testData[i][0], testData[i][1]
		preData = np.zeros((time_step, n_inputs))
		for j in range(len(data)):
				if j>0 and j%2==0:
						preData[data[j-1]][data[j-2]]=1
		
		test_data.append(preData)
		test_labels.append(label)

test_data=np.reshape(test_data, [test_size//n_batch, n_batch, time_step, n_inputs])
test_labels=np.reshape(test_labels, [test_size//n_batch, n_batch, ])

#ALIF############################################
cell=ALIF(seed, n_inputs, n_hidden, n_outputs, \
t_ref, tau_m, tau_w, Vth, beta, tau_syn, tau_post, Vmax)

#Build network
inputs = tf.placeholder(tf.float32, [None, time_step, n_inputs], name='inputs')
targets = tf.placeholder(tf.int64, [None, ], name='Targets')

init_state = cell.zero_state(n_batch, tf.float32)
rnn_out, states = tf.nn.dynamic_rnn(cell, inputs, initial_state=init_state)
o1, v, z, b, _, in_cur, syn_cur, win, wrec = rnn_out
av = tf.reduce_mean(z, axis=(0, 1))

rd.seed(seed)
w_out = tf.Variable(rd.randn(n_hidden, n_outputs) / np.sqrt(int(n_hidden)), dtype=tf.float32, name='out_weight')
b_out = tf.Variable(rd.randn(n_outputs), dtype=tf.float32, name='out_bias')

o = tf.matmul(o1, w_out) + b_out
o = o[:, -1, :]

Y_predict_num = tf.argmax(o, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(targets, Y_predict_num), dtype=tf.float32))
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=o))

learning_rate = tf.Variable(learning_rate, dtype=tf.float32, trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
train_step = optimizer.minimize(loss=loss, global_step=global_step)

plot_result={ 
'in_cur': in_cur, 
'syn_cur': syn_cur,
'adp_cur': b, 
'win': win,
'wrec': wrec,
'spike': z,
'av': av,
'voltage': v,
'rec_out': o1,
'targetNum':targets,
'input_spikes': inputs,
}


#Functions######################################
def get_stats(v, flag):
		if flag:
				k_min = np.sum(v == np.min(v))
				k_max = np.sum(v == np.max(v))
				return np.mean(v), np.std(v), np.min(v), np.max(v), k_min, k_max
		else:
				return np.mean(v), np.std(v), np.min(v), np.max(v)

def print_val(presult):
		print("win:", get_stats(presult['win'], 0))
		print("wrec:", get_stats(presult['wrec'], 0))
		print("in_cur:", get_stats(presult['in_cur'], 0))
		print("syn_cur:", get_stats(presult['syn_cur'], 0))
		print("adp_cur:", get_stats(presult['adp_cur'], 0))
		print("spike:", get_stats(presult['av']*1000, 1))

def store_val(presult):
		name=foldername+"/"+filename
		np.save(name+"_wrec_"+str(e), presult['wrec'])
		np.save(name+"_win_"+str(e), presult['win'])
		np.save(name+"_target_"+str(e),presult['targetNum'])
		np.save(name+"_pixel_"+str(e),presult['input_spikes'])

#Run##############################################
session = tf.Session()
session.run(tf.global_variables_initializer())

for e in range(epoch):

		#check initial status###############
		if e==0:
				train_dict={inputs: train_data[0], targets: train_labels[0]}
				presult = session.run(plot_result, feed_dict=train_dict)
				print_val(presult)
				store_val(presult)
		
		#training###########################
		total_acc=0.0
		for i in range(train_size//n_batch):
				
				train_dict={inputs: train_data[i], targets: train_labels[i]}
				presult, acc,  _ = session.run([ \
				plot_result, accuracy, train_step], feed_dict=train_dict)
				
				total_acc+=acc

				if e%10==0 and i==train_size//n_batch-1:
						print_val(presult)

		print(e, "training:", total_acc/len(train_data))
		
		#testing############################
		total_acc=0.0
		for j in range(test_size//n_batch):
				test_dict={inputs: test_data[j], targets: test_labels[j]}
				acc=session.run(accuracy, feed_dict=test_dict)
				total_acc+=acc
		print(e, "testing:", total_acc/len(test_data))



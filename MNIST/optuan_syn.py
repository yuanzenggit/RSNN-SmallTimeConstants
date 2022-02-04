#System print##################################
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import optuna

#Libraries####################################
import numpy as np
import random
from model_synapse import ALIF
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy.random as rd
import sys
import input_data_custom

def objective(trial):
		
		#Parameters####################################
		seed=1
		train_size = 55000  
		test_size = 10000 
		valid_size = 5000
		time_step = 784
		n_classes = 10
		n_inputs = 1
		n_hidden = 200 
		n_outputs = n_classes

		n_batch = 256
		learning_rate = 0.1
		epoch = 1

		t_ref=1
		tau_m=trial.suggest_int('tau_m', 1, 701, step=50) 
		tau_w=trial.suggest_int('tau_w', 1, 701, step=50)
		Vth=trial.suggest_uniform('thr', 0.01, 10)
		beta=trial.suggest_int('beta', 1, 100, step=10)
		tau_syn=trial.suggest_int('tau_syn', 1, 701, step=50) 
		Vmax=trial.suggest_float('vmax', 1.1, 2, step=0.1)
		tau_post=trial.suggest_int('tau_p', 1, 701, step=50) 

		print("seed:", seed, "train_size:", train_size, "test_size", test_size, \
		"time_step:", time_step, "network:", n_inputs, n_hidden, n_classes,\
		"n_batch:", n_batch, "lr:", learning_rate)
		print("ref:",t_ref, "tau_m:", tau_m, "tau_w:", tau_w, \
		"Vth:", Vth, "beta:", beta, "tau_syn:", tau_syn, "tau_post", tau_post, "Vmax:", Vmax)

		#Read in Ti_digit dataset######################
		mnist = input_data_custom.read_data_sets('MNIST_data', one_hot=True,
		train_size=train_size, test_size=test_size, validation_size=valid_size)
		print(len(mnist.train.images), len(mnist.test.images), len(mnist.validation.images))

		def get_data_dict(batch_size, type='train'):
				if type == 'test':
						input_px, target = mnist.test.next_batch(batch_size, shuffle=False)
				elif type == 'validation':
						input_px, target = mnist.validation.next_batch(batch_size, shuffle=False)
				elif type == 'train':
						input_px, target = mnist.train.next_batch(batch_size, shuffle=False)
				target_num = np.argmax(target, axis=1)
				input_px = np.expand_dims(input_px, axis=2)
				data_dict = {inputs: input_px, targets: target_num}

				return data_dict

		#ALIF############################################
		cell=ALIF(seed, n_inputs, n_hidden, n_outputs, \
		t_ref, tau_m, tau_w, Vth, beta, tau_syn, tau_post, Vmax)

		#Build network
		inputs = tf.placeholder(dtype=tf.float32, shape=(None, None, n_inputs),name='InputSpikes')
		targets = tf.placeholder(dtype=tf.int64, shape=(None,),name='Targets')

		init_state = cell.zero_state(n_batch, tf.float32)
		rnn_out, states = tf.nn.dynamic_rnn(cell, inputs, initial_state=init_state)
		o1, v, z, b, _, in_cur, syn_cur, win, wrec = rnn_out
		av = tf.reduce_mean(z, axis=(0, 1))

		rd.seed(seed)
		w_out = tf.Variable(rd.randn(n_hidden, n_outputs) / np.sqrt(200), dtype=tf.float32, name='out_weight')
		b_out = tf.Variable(rd.randn(n_outputs), dtype=tf.float32, name='out_bias')

		o = tf.matmul(o1, w_out) + b_out
		o = o[:, -1, :]

		#Loss
		Y_predict_num = tf.argmax(o, axis=1)
		accuracy = tf.reduce_mean(tf.cast(tf.equal(targets, Y_predict_num), dtype=tf.float32))
		loss_recall = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=o))
		loss=loss_recall
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
		'o': b_out,
		}

		def get_stats(v, flag):
				if flag:
						k_min = np.sum(v == np.min(v))
						k_max = np.sum(v == np.max(v))
						return np.min(v), np.max(v), np.mean(v), np.std(v), k_min, k_max
				else:
						return np.mean(v), np.std(v), np.min(v), np.max(v)

		#Run##############################################
		session = tf.Session(config=config)
		session.run(tf.global_variables_initializer())

		for e in range(epoch):

				print("iter.....", e)
				#train
				train_acc=0.0
				for i in range(train_size//n_batch):
						train_dict=get_data_dict(n_batch, type='train')
						acc,  _ = session.run([accuracy, train_step], feed_dict=train_dict)
						train_acc+=acc
				
				#validation
				valid_acc=0.0
				valid_dict=get_data_dict(n_batch, type='validation')
				presult, acc, _ = session.run([plot_result, accuracy, train_step], feed_dict=valid_dict)
				valid_acc+=acc

				if e%10==0:
					print("win:", get_stats(presult['win'], 0))
					print("wrec:", get_stats(presult['wrec'], 0))
					print("in_cur:", get_stats(presult['in_cur'], 0))
					print("syn_cur:", get_stats(presult['syn_cur'], 0)) 
					print("adp_cur:", get_stats(presult['adp_cur'], 0))
					print("spike:", get_stats(presult['av']*1000, 1))

				print(e, "validation:", round(valid_acc,3))

				#total_acc=0.0
				#for j in range(test_size//n_batch):
				#	test_dict={inputs: test_data[j], targets: test_labels[j]}
				#		acc=session.run(accuracy, feed_dict=test_dict)
				#		total_acc+=acc
				#print e, "testing:", total_acc/len(test_data)

		return valid_acc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)

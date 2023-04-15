"""
author: KhanhTN - Cre: LyVT
"""
import tensorflow as tf
import os
import numpy as np
# from tool import *
import time
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from tool import *

class Model(object):
	"""
	This class is used to train the url phishing detection models step by step based on NLP techniques.

	"""
	def __init__(self, labId, model_type = 'lstm', embed_mode="glove", train_data_dir = '', val_size = 0.1):
		"""
		Init class 
		Parameters:
		-----------
		labId: 	string
				This is used to define the lab. Using for backend.

		model_type:	string, default = 'lstm'
					Type of RNN cell. Including: lstm, gru, bilstm.

		embed_mode: string, default = 'glove'      
					Type of embedding model.  Including: glove, word2vec

		train_data_dir: string, default = ''
					Directory path of training data. This directory includes two sub-directories, i.e., phishing and normal
		
		val_size: float, default = 0.1
					Size of validation dataset over all training dataset.
		"""
		#Get training data, tokenizer and embeding matrix
		self.trainX, self.valX, self.trainY, self.valY, self.tokenizer = get_train_data(train_data_dir, val_size, embed_mode=embed_mode)
		self.embedding_matrix = get_embedding_vectors(self.tokenizer,embed_mode=embed_mode)
		# create model
		self.model = get_model(tokenizer=self.tokenizer, embedding_matrix = self.embedding_matrix, rnn_cell= model_type, embed_mode=embed_mode)
		self.optimizer = tf.keras.optimizers.Adam(lr=0.001)
		# Create directory to save training models
		self.checkpoint_dir = f'./modelDir/{labId}/log_train/{model_type}'
		if not os.path.exists(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
		self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,model=self.model)

		#save tokenizer to file
		output = open(os.path.join (self.checkpoint_dir,'tokenizer.pkl'), 'wb')
		pickle.dump(self.tokenizer, output)
		output.close()
		#save embeding matrix to file
		output = open(os.path.join (self.checkpoint_dir,'embedding_matrix.pkl'), 'wb')
		pickle.dump(self.embedding_matrix, output)
		output.close()
 
	def step(self, X, y):
		"""
		Train one batch
		Parameters:
		-----------
		X: array like, shape (n_samples, n_features)
			Input data

		y: array like, shape (n_samples, n_classes)
			Input label

		Returns:
		-------
		trainable variables for training model
		"""
		with tf.GradientTape() as tape:
			pred = self.model(X) # make a prediction using the model and then calculate the loss
			loss = tf.keras.losses.categorical_crossentropy(y, pred)
		grads = tape.gradient(loss, self.model.trainable_variables) # calculate the gradients using our tape and then update the model weights
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

	def train (self, learning_rate = 0.001, EPOCHS = 10, BS = 32):
		"""
		Training data step by step

		Parameters:
		-----------
		learning_rate: float, default = 0.001
						Learning rate for training

		EPOCHS: int, default = 10
				Number of training epochs

		BS: int, default = 32
			Batch size of training data

		Returns:
		--------
		train_acc: accuracy of training dataset for each training epoch for backend

		val_acc: accuracy of validation dataset for each training epoch for backend

		model: trained model
		"""
		self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate, decay=learning_rate / EPOCHS)
		numUpdates = int(self.trainX.shape[0] / BS) # compute the number of batch updates per epoch
		for epoch in range(0, EPOCHS):
			print("[INFO] starting epoch {}/{}...".format(epoch + 1, EPOCHS), end="")
			epochStart = time.time()
			for i in range(0, numUpdates):
				start = i * BS
				end = start + BS
				self.step(self.trainX[start:end], self.trainY[start:end])
			epochEnd = time.time()
			elapsed = (epochEnd - epochStart) / 60.0
			print("took {:.4} minutes".format(elapsed))

			#Calculating Train loss acc
			train_acc = self.evaluate (self.trainX, self.trainY)
			val_acc = self.evaluate (self.valX, self.valY)
			print (train_acc, val_acc)
			#save to checkpoint
			self.checkpoint.save(file_prefix = self.checkpoint_prefix)
			#yield value for backend
			# yield
			print({
					"train_acc": train_acc,
					"val_acc": val_acc
				}) 

	def evaluate(self, batch_input, batch_target):
		"""
		Evaluating trained models
		Parameters:
		-----------
		batch_input: array like, shape (n_samples, n_features)
			Input data

		batch_target: array like, shape (n_samples, n_classes)
			Input label

		Returns:
		-------
		Accuracy of the model
		"""

		current_batch_size = batch_input.shape[0]
		batch_prediction = self.model(batch_input)
		batch_prediction = tf.argmax(batch_prediction, axis = 1)
		batch_target = tf.argmax(batch_target, axis = 1)
		batch_true  = np.equal(batch_prediction, batch_target)
		batch_true = np.sum(batch_true)
		return batch_true / current_batch_size
# from model import Model
import tensorflow as tf
import os
import pickle
from tool import get_model, get_test_data
# import pdb
import numpy as np

def test(test_data_dir, labId, ckpt_number, model_type, embed_mode):
	"""
	Testing trained models.

	Parameters:
	----------
	test_data_dir:	string,
					Directory path of testing data.

	labId: 	string,
			ID of lab.

	ckpt_number: 	int,
					Number of checkpoint model for testing.

	model_type: 	string,
					Type of rnn cell model.

	embed_mode: string		
			Type of embedding model. 
	Returns:
	--------
	Accuracy of testing model on the testing dataset.

	"""

	test_acc = 0
	#Model directory
	model_dir = f'./modelDir/{labId}/log_train/{model_type}'

	#Get tokenizer from file
	tokenizer_file = open(os.path.join (model_dir,'tokenizer.pkl'), 'rb')
	tokenizer = pickle.load(tokenizer_file)
	tokenizer_file.close()

	#get embeding matrix from file
	embeding_matrix_file = open(os.path.join (model_dir,'embedding_matrix.pkl'), 'rb')
	embeding_matrix = pickle.load(embeding_matrix_file)
	embeding_matrix_file.close()

	#Checkpoint path
	ckpt_path = os.path.join (model_dir, 'ckpt-'+str (ckpt_number))

	#Get model and load model from checkpoint path
	model = get_model(tokenizer=tokenizer, embedding_matrix = embeding_matrix, rnn_cell= model_type, embed_mode=embed_mode)
	checkpoint = tf.train.Checkpoint(model = model)
	checkpoint.restore (ckpt_path)

	#Get testing data
	testX, testY = get_test_data(test_data_dir, tokenizer, embed_mode)
	number_sample = testX.shape[0]

	#Inference testing data and compute accuracy
	prediction = model(testX)
	prediction = tf.argmax(prediction, axis = 1)
	target = tf.argmax(testY, axis = 1)
	batch_true  = np.equal(prediction, target)
	batch_true = np.sum(batch_true)
	test_acc = batch_true / number_sample

	#yield for backend.

	print({
			"test_acc": test_acc
		  })

	


if __name__ == '__main__':
	test('data/test.csv', 'lab3', 1, 'lstm', "word2vec")
	
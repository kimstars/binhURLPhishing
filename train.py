from model import Model


def main(data_dir,learning_rate, epochs, batch_size, val_size, model_type, embed_mode, labId):
	"""
	Train model

	Parameters:
	-----------
	data_dir: 	str,
				Training data directory.

	learning_rate: 	float,
					Learning rate for training model.
	epochs:	int,
			Number of training epochs.

	batch_size: int,
				Batch size of training data.
	val_size: 	float,
				Size of validation set over training dataset
	model_type: string,
				Type of rnn cells for building model

	labId:	string,
			ID of lab (use for backend)
	Returns:
	--------
	Trained models saved by .ckpt file
	"""
    
    #Call model from Model class for training
	model = Model(labId, model_type, train_data_dir = data_dir, val_size=val_size,embed_mode=embed_mode)
	model.train(learning_rate, epochs, batch_size)
	# print(result)

if __name__ == '__main__':
	main("data/train.csv", 0.001, 4, 8, 0.2, 'lstm', 'word2vec','lab4')

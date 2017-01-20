from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import numpy as np

import configuration as config
import models


class PreProcessing:
		
	def loadData(self):   
		print "loading data..."
		data_src = config.data_src
		text_pos = open(data_src[0],"r").readlines()
		text_neg = open(data_src[1],"r").readlines()
		labels_pos = [1]*len(text_pos)
		labels_neg = [0] * len(text_neg)
		texts = text_pos
		texts.extend(text_neg)
		labels = labels_pos
		labels.extend(labels_neg)
		
		   
		tokenizer = Tokenizer(nb_words=config.MAX_NB_WORDS)
		tokenizer.fit_on_texts(texts)
		sequences = tokenizer.texts_to_sequences(texts)

		word_index = tokenizer.word_index
		print('Found %s unique tokens.' % len(word_index))

		data = pad_sequences(sequences, maxlen=config.MAX_SEQUENCE_LENGTH)

		labels = np_utils.to_categorical(np.asarray(labels))
		print('Shape of data tensor:', data.shape)
		print('Shape of label tensor:', labels.shape)

		# split the data into a training set and a validation set
		indices = np.arange(data.shape[0])
		np.random.shuffle(indices)
		data = data[indices]
		labels = labels[indices]
		nb_validation_samples = int(config.VALIDATION_SPLIT * data.shape[0])

		self.x_train = data[:-nb_validation_samples]
		self.y_train = labels[:-nb_validation_samples]
		self.x_val = data[-nb_validation_samples:]
		self.y_val = labels[-nb_validation_samples:]
		self.word_index = word_index

	def loadEmbeddings(self):
		embeddings_src = config.embeddings_src
		word_index = self.word_index
		embeddings_index = {}
		f = open(embeddings_src)
		i=0
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
			EMBEDDING_DIM = len(coefs)
			i+=1
			if i>10000:
				break
		f.close()

		print('Found %s word vectors.' % len(embeddings_index))
		embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
		for word, i in word_index.items():
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				# words not found in embedding index will be all-zeros.
				embedding_matrix[i] = embedding_vector
		self.embedding_matrix = embedding_matrix
		self.EMBEDDING_DIM = EMBEDDING_DIM
		self.MAX_SEQUENCE_LENGTH = config.MAX_SEQUENCE_LENGTH
		

def main():
	preprocessing = PreProcessing()
	preprocessing.loadData()
	preprocessing.loadEmbeddings()
	
	cnn_model = models.CNNModel()
	params_obj = config.Params()
	
	# Establish params
	params_obj.num_classes=2
	params_obj.vocab_size = len(preprocessing.word_index)
	params_obj.inp_length = preprocessing.MAX_SEQUENCE_LENGTH
	params_obj.embeddings_dim = preprocessing.EMBEDDING_DIM
            
	# get model
	model = cnn_model.getModel(params_obj=params_obj, weight=preprocessing.embedding_matrix)
	
	x_train, y_train, x_val, y_val = preprocessing.x_train, preprocessing.y_train, preprocessing.x_val, preprocessing.y_val
	# train
	model.fit(x_train, y_train, validation_data=(x_val, y_val),
              nb_epoch=params_obj.num_epochs, batch_size=params_obj.batch_size)
              
	#evaluate
	

if __name__ == "__main__":
	main()

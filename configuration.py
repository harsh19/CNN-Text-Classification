data_src = ["./data/rt-polarity.pos","./data/rt-polarity.neg"] 
embeddings_src = "./data/glove.6B.300d.txt"
MAX_NB_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100
VALIDATION_SPLIT = 0.1

####################################
class Params:
	inp_len = None
	vocab_size = None
	num_classes= None
	
	use_pretrained_embeddings = True
	embeddings_dim = 300
	train_embedding = True
	filter_sizes = [3,4,5]
	filter_numbers = [100,100,100]
	filter_pool_lengths = [2,2,2]
	num_epochs = 2
	batch_size = 50
	dropout_val = 0.5
	dense_layer_size = 100
	num_output_classes = 2
	use_two_channels = True
	
	def setParams(self, dct):
		if 'vocab_size' in dct:
			self.vocab_size=dct['vocab_size']
		if 'embeddings_size' in dct:
			self.embeddings_size=dct['embeddings_size']
		if 'inp_len' in dct:
			self.inp_len=dct['inp_len']
		if 'train_embedding' in dct:
			self.train_embedding=dct['train_embedding']
		if 'filter_sizes' in dct:
			self.filter_sizes=dct['filter_sizes']
		if 'filter_numbers' in dct:
			self.filter_numbers=dct['filter_numbers']			
		if 'filter_pool_lengths' in dct:
			self.filter_pool_lengths=dct['filter_pool_lengths']

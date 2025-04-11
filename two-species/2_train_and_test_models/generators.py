'''
	This file implements classes that extend Keras's Sequence class
	as iteratable data loaders for model training and validation.

	These generators use the PyfaidxCoordsToVals functionality from
	the seqdataloader package (https://github.com/kundajelab/seqdataloader),
	which uses a fasta index to quickly generate one-hot encodings of
	sequences from a bed-formatted coordinate set corresponding to a
	region in the genome. Because seqdataloader relies on a fasta index,
	it is necessary for your genome fasta files (the paths of which are
	specified in default_params.py) to have accompanying fasta indexes
	in the same directory. These can be made using the pyfaidx package:

		import pyfaidx
		Fasta(genome_fasta_filepath)

	The generators extract dataset filepaths and other parameters from
	the Params object given as input. During initialization, they load in
	the coordinates from their input bed files. These coordinates correspond
	to examples that the generator will iteratively return. When a batch
	of examples is fetched from the generator, that batch's worth of
	coordinates is converted to one-hot sequences using the
	PyfaidxCoordsToVals object. Labels for the examples are also produced
	in the gase of the training generators.

	Because parts of the training datasets are epoch-specific, the
	coordinate sets for those parts of the data must be re-loaded at the
	end of each epoch. Thus, the training generators also implement an
	on_epoch_end method that updates the filepaths for the epoch-specific
	data and re-loads the coordinate sets from those files.
'''

import numpy as np
import random
import tensorflow as tf

from params import ROOT, GENOMES
from keras.utils import Sequence # alias for PyDataset!
from seqdataloader.batchproducers.coordbased.core import Coordinates
from seqdataloader.batchproducers.coordbased.coordstovals.fasta import PyfaidxCoordsToVals

class TrainGenerator_SingleTask(Sequence):
	'''
		Alias for keras.utils.PyDataset! 
		- https://www.tensorflow.org/api_docs/python/tf/keras/utils/PyDataset

		This class implements an iterable data loader for the basic model
		(non-domain-adaptive) for use during training. The training data
		consists of bound (positive) and unbound (negative) examples. This
		class loads in coordinates from the bound examples file and the
		unbound examples file separately, and then when a batch is accessed,
		it combines 50% bound examples and 50% unbound examples, with
		corresponding labels of 1 and 0, into a batch's worth of one-hot
		encoded sequences.
		
		Because the unbound examples are different every epoch, those
		example coordinates are re-loaded from a new file at the end of
		each epoch.
	'''
	def __init__(self, params, workers=1, use_multiprocessing=False, max_queue_size=10):
		super().__init__(workers=workers, use_multiprocessing=use_multiprocessing, max_queue_size=max_queue_size)
		self.params				= params
		self.posfile			= params.bindingtrainposfile
		self.negfile			= params.bindingtrainnegfile
		self.converter			= PyfaidxCoordsToVals(params.source_genome_file)
		self.batchsize			= params.batchsize
		self.halfbatchsize		= self.batchsize // 2
		self.steps_per_epoch	= params.train_steps
		self.total_epochs		= params.epochs
		self.current_epoch		= 0

		self.get_data()

	def __len__(self):
		return self.steps_per_epoch

	def __getitem__(self, batch_index):
		# (1) Get chunk of onehots
		pos_onehots_batch = self.pos_onehots[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize]
		neg_onehots_batch = self.neg_onehots[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize]
		assert pos_onehots_batch.shape[0] > 0, pos_onehots_batch.shape[0]
		assert neg_onehots_batch.shape[0] > 0, neg_onehots_batch.shape[0]

		# (2) We combine bound and unbound sites into one large array, and create 
		# label vector. We don't need to shuffle here because all these examples will
		# correspond to a simultaneous gradient update for the whole batch.
		all_seqs = np.concatenate((pos_onehots_batch, neg_onehots_batch))

		# Labels for binding prediction task
		binding_labels = np.concatenate((
			np.ones(pos_onehots_batch.shape[0],),
			np.zeros(neg_onehots_batch.shape[0],)
		))

		# Auxillary variable representing the domain index (species) for each example
		domain_labels = np.concatenate((
			np.ones(pos_onehots_batch.shape[0],) * self.params.domain_indices[self.params.source_species],
			np.ones(neg_onehots_batch.shape[0],) * self.params.domain_indices[self.params.source_species]
		))

		assert all_seqs.shape[0] == self.batchsize, all_seqs.shape[0]

		return {"sequence":all_seqs, "index":domain_labels}, binding_labels

	def get_data(self):
		''' Using current filenames stored in self.posfile and self.negfile,
			load in all of the training data as coordinates only.
			Then, when it is time to fetch individual batches, a chunk of
			coordinates will be converted into one-hot encoded sequences
			ready for model input.
		'''
		with open(self.posfile) as posf:
			pos_coords_tmp		= [line.split()[:3] for line in posf]  # expecting bed file format
			self.pos_coords		= [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in pos_coords_tmp]  # no strand consideration
			self.pos_onehots	= self.converter(self.pos_coords)

		with open(self.negfile) as negf:
			neg_coords_tmp		= [line.split()[:3] for line in negf]
			self.neg_coords		= [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in neg_coords_tmp]
			self.neg_onehots	= self.converter(self.neg_coords)

	def on_epoch_end(self):
		print(f"\n\nFinished training on the set (neg): {self.negfile}!\n")

		# (1) Switch to next set of negative examples
		prev_epoch = self.current_epoch
		self.current_epoch = prev_epoch + 1

		# (2) Update file where we will retrieve unbound site coordinates from
		prev_negfile = self.negfile
		next_negfile = prev_negfile.replace(str(prev_epoch) + "E", str(self.current_epoch) + "E")
		self.negfile = next_negfile

		if self.total_epochs < self.current_epoch:
			return
		else:
			# (3) Load in new unbound site coordinates
			with open(self.negfile) as negf:
				neg_coords_tmp = [line.split()[:3] for line in negf]
				self.neg_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in neg_coords_tmp]
				self.neg_onehots = self.converter(self.neg_coords)
		
			# (4) Then shuffle positive examples
			self.pos_onehots = tf.random.shuffle(self.pos_onehots)

class TrainGenerator_Adaptive(Sequence):
	def __init__(self, params, workers=1, use_multiprocessing=False, max_queue_size=10):
		super().__init__(workers=workers, use_multiprocessing=use_multiprocessing, max_queue_size=max_queue_size)
		self.params = params

		# Binding task
		self.posfile = params.bindingtrainposfile
		self.negfile = params.bindingtrainnegfile

		# Domain-adaptive task
		self.source_species_file = params.source_species_file
		self.target_species_file = params.target_species_file
		
		# Converters for both species since we have mutli-task learning
		self.source_converter = PyfaidxCoordsToVals(params.source_genome_file)
		self.target_converter = PyfaidxCoordsToVals(params.target_genome_file)

		# Other parameters we need
		self.batchsize			= params.batchsize
		self.halfbatchsize		= self.batchsize // 2
		self.steps_per_epoch	= params.train_steps
		self.total_epochs		= params.epochs
		self.current_epoch		= 0

		self.get_binding_data()
		self.get_species_data()

	def __len__(self):
		return self.steps_per_epoch

	def __getitem__(self, batch_index):

		# (1) Retrieve a chunk of coordinates for both the bound and unbound site examples,
		# and convert those coordinates to one-hot encoded sequence arrays
		pos_onehot_batch = self.pos_onehots[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize]
		neg_onehot_batch = self.neg_onehots[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize]
		assert pos_onehot_batch.shape[0] > 0, pos_onehot_batch.shape[0]
		assert neg_onehot_batch.shape[0] > 0, neg_onehot_batch.shape[0]

		# (2) We do the same thing again, but for the "species-background" data of both species
		source_onehot_batch = self.source_onehots[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize]
		target_onehot_batch = self.target_onehots[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize]
		assert source_onehot_batch.shape[0] > 0, source_onehot_batch.shape[0]
		assert target_onehot_batch.shape[0] > 0, target_onehot_batch.shape[0]
		
		# (3) Concatenate all the one-hot encoded sequences together
		all_seqs = np.concatenate((pos_onehot_batch, neg_onehot_batch, source_onehot_batch, target_onehot_batch))

		# (4) Create label vectors for both tasks
		
		# NOTE: The label of -1 will correspond to a masking of the loss function
		# (so if the label for the binding task is -1 for example i, then when the
		# loss gradient backpropagates, example i will not be included in that calculation

		# Label for binding prediction task:
		binding_labels = np.concatenate((
			np.ones(pos_onehot_batch.shape[0],),
			np.zeros(neg_onehot_batch.shape[0],),
			-1 * np.ones(source_onehot_batch.shape[0],),
			-1 * np.ones(target_onehot_batch.shape[0],)
		))

		# Label for domain-adaptive task:
		species_labels = np.concatenate((
			-1 * np.ones(pos_onehot_batch.shape[0],),
			-1 * np.ones(neg_onehot_batch.shape[0],),
			np.ones(source_onehot_batch.shape[0],) * self.params.domain_indices[self.params.source_species],
			np.ones(target_onehot_batch.shape[0],) * self.params.domain_indices[self.params.target_species]
		))

		# Auxillary variable representing the domain index (species) for each example
		domain_labels = np.concatenate((
			np.ones(pos_onehot_batch.shape[0],) * self.params.domain_indices[self.params.source_species],
			np.ones(neg_onehot_batch.shape[0],) * self.params.domain_indices[self.params.source_species],
			np.ones(source_onehot_batch.shape[0],) * self.params.domain_indices[self.params.source_species],
			np.ones(target_onehot_batch.shape[0],) * self.params.domain_indices[self.params.target_species]
		))

		assert all_seqs.shape[0] == self.batchsize * 2, all_seqs.shape[0]

		# Everything should have the same len!
		assert binding_labels.shape == species_labels.shape, (binding_labels.shape, species_labels.shape)
		assert species_labels.shape == domain_labels.shape, (species_labels.shape, domain_labels.shape)

		# Here we assign the name "classifier" to the binding prediction task, and "discriminator"
		return {"sequence":all_seqs, "index":domain_labels}, binding_labels

	def get_binding_data(self):
		'''
			Using current filenames stored in self.posfile and self.negfile,
			load in all of the "binding" training data as coordinates only.
			Then, when it is time to fetch individual batches, a chunk of
			coordinates will be converted into one-hot encoded sequences
			ready for model input.

			All data here (the binding task) comes from the source domain ONLY.
			That is consistent with the UDA setting :)
		'''
		with open(self.posfile) as posf:
			pos_coords_tmp		= [line.split()[:3] for line in posf]
			self.pos_coords		= [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in pos_coords_tmp]
			self.pos_onehots	= self.source_converter(self.pos_coords)

		with open(self.negfile) as negf:
			neg_coords_tmp		= [line.split()[:3] for line in negf]
			self.neg_coords		= [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in neg_coords_tmp]
			self.neg_onehots	= self.source_converter(self.neg_coords)

	def get_species_data(self):
		'''
			Using current filenames stored in self.source_species_file
			and self.target_species_file. There are the "species-background"
			training data for both species. These examples are used to inform
			whatever domain-adaptive task we go with.

			All data here (the domain-adaptive task) comes from the BOTH domains.
		'''
		with open(self.source_species_file) as sourcef:
			source_coords_tmp	= [line.split()[:3] for line in sourcef]
			self.source_coords	= [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in source_coords_tmp]
			self.source_onehots	= self.source_converter(self.source_coords)

		with open(self.target_species_file) as targetf:
			target_coords_tmp	= [line.split()[:3] for line in targetf]
			self.target_coords	= [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in target_coords_tmp]
			self.target_onehots	= self.target_converter(self.target_coords)

	def on_epoch_end(self):
		print(f"\n\nFinished training on the set (binding, neg): {self.negfile}\n")
		print(f"\nFinished training on the set (source, background): {self.source_species_file}\n")
		print(f"\nFinished training on the set (target, background): {self.target_species_file}\n\n")

		# (1) Switch to next set of negative examples
		prev_epoch = self.current_epoch
		self.current_epoch = prev_epoch + 1

		# (2) Update files to pull data from, for unbound examples and species-background examples
		prev_negfile = self.negfile
		next_negfile = prev_negfile.replace(str(prev_epoch) + "E", str(self.current_epoch) + "E")
		self.negfile = next_negfile

		if self.total_epochs < self.current_epoch:
			return
		else:
			prev_sourcefile = self.source_species_file
			next_sourcefile = prev_sourcefile.replace(str(prev_epoch) + "E", str(self.current_epoch) + "E")
			self.source_species_file = next_sourcefile

			prev_targetfile = self.target_species_file
			next_targetfile = prev_targetfile.replace(str(prev_epoch) + "E", str(self.current_epoch) + "E")
			self.target_species_file = next_targetfile

			# (3) Load new data into memory for unbound examples & species-background examples		
			with open(self.negfile) as negf:
				neg_coords_tmp = [line.split()[:3] for line in negf]
				self.neg_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in neg_coords_tmp]
				self.neg_onehots = self.source_converter(self.neg_coords)

			with open(self.source_species_file) as sourcef:
				source_coords_tmp = [line.split()[:3] for line in sourcef]
				self.source_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in source_coords_tmp]
				self.source_onehots = self.source_converter(self.source_coords)

			with open(self.target_species_file) as targetf:
				target_coords_tmp = [line.split()[:3] for line in targetf]
				self.target_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in target_coords_tmp]
				self.target_onehots = self.target_converter(self.target_coords)

			# (4) Then shuffle positive examples
			self.pos_onehots = tf.random.shuffle(self.pos_onehots)

class TrainGenerator_MultiTask(Sequence):
	'''
		This class implements an iterable data loader for the domain-adaptive
		model for use during training. Part 1 of the training data
		consists of bound (positive) and unbound (negative) examples. This
		class loads in coordinates from the bound examples file and the
		unbound examples file separately, and then when a batch is accessed,
		it combines 50% bound examples and 50% unbound examples, with
		corresponding binding labels of 1 and 0, into the first half of the
		batch's worth of one-hot encoded sequences.
		
		Part 2 of the training data consists of "species-background" examples
		originating from both the source and target species. This generator
		loads in coordinates for examples from each species separately from
		the binding examples in Part 1. When a batch is accessed, the second
		half of that batch is produced by transforming those coordinates to
		their one-hot encoded sequences. These examples have "masked" binding
		labels (-1) which will cause them to be disregarded in the loss
		calculations for the binding classifier head of the model.
		
		For both parts of each batch, "species discriminator" task labels are
		also produced, but for the binding classifier training examples, the
		species labels are masked (-1), so that the binding classifier never
		trains on examples from the target species.
		
		Because the unbound and species-backgroundexamples are different every
		epoch, those coordinates are re-loaded from new files at the end of
		each epoch.
	'''
	def __init__(self, params, workers=1, use_multiprocessing=False, max_queue_size=10):
		super().__init__(workers=workers, use_multiprocessing=use_multiprocessing, max_queue_size=max_queue_size)
		self.params = params

		# Binding task
		self.posfile = params.bindingtrainposfile
		self.negfile = params.bindingtrainnegfile

		# Domain-adaptive task
		self.source_species_file = params.source_species_file
		self.target_species_file = params.target_species_file
		
		# Converters for both species since we have mutli-task learning
		self.source_converter = PyfaidxCoordsToVals(params.source_genome_file)
		self.target_converter = PyfaidxCoordsToVals(params.target_genome_file)

		# Other parameters we need
		self.batchsize			= params.batchsize
		self.halfbatchsize		= self.batchsize // 2
		self.steps_per_epoch	= params.train_steps
		self.total_epochs		= params.epochs
		self.current_epoch		= 0

		self.get_binding_data()
		self.get_species_data()

	def __len__(self):
		return self.steps_per_epoch

	def __getitem__(self, batch_index):

		# (1) Retrieve a chunk of coordinates for both the bound and unbound site examples,
		# and convert those coordinates to one-hot encoded sequence arrays
		pos_onehot_batch = self.pos_onehots[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize]
		neg_onehot_batch = self.neg_onehots[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize]
		assert pos_onehot_batch.shape[0] > 0, pos_onehot_batch.shape[0]
		assert neg_onehot_batch.shape[0] > 0, neg_onehot_batch.shape[0]

		# (2) We do the same thing again, but for the "species-background" data of both species
		source_onehot_batch = self.source_onehots[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize]
		target_onehot_batch = self.target_onehots[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize]
		assert source_onehot_batch.shape[0] > 0, source_onehot_batch.shape[0]
		assert target_onehot_batch.shape[0] > 0, target_onehot_batch.shape[0]
		
		# (3) Concatenate all the one-hot encoded sequences together
		all_seqs = np.concatenate((pos_onehot_batch, neg_onehot_batch, source_onehot_batch, target_onehot_batch))

		# (4) Create label vectors for both tasks
		
		# NOTE: The label of -1 will correspond to a masking of the loss function
		# (so if the label for the binding task is -1 for example i, then when the
		# loss gradient backpropagates, example i will not be included in that calculation

		# Label for binding prediction task:
		binding_labels = np.concatenate((
			np.ones(pos_onehot_batch.shape[0],),
			np.zeros(neg_onehot_batch.shape[0],),
			-1 * np.ones(source_onehot_batch.shape[0],),
			-1 * np.ones(target_onehot_batch.shape[0],)
		))

		# Label for domain-adaptive task:
		species_labels = np.concatenate((
			-1 * np.ones(pos_onehot_batch.shape[0],),
			-1 * np.ones(neg_onehot_batch.shape[0],),
			np.ones(source_onehot_batch.shape[0],) * self.params.domain_indices[self.params.source_species],
			np.ones(target_onehot_batch.shape[0],) * self.params.domain_indices[self.params.target_species]
		))

		# Auxillary variable representing the domain index (species) for each example
		domain_labels = np.concatenate((
			np.ones(pos_onehot_batch.shape[0],) * self.params.domain_indices[self.params.source_species],
			np.ones(neg_onehot_batch.shape[0],) * self.params.domain_indices[self.params.source_species],
			np.ones(source_onehot_batch.shape[0],) * self.params.domain_indices[self.params.source_species],
			np.ones(target_onehot_batch.shape[0],) * self.params.domain_indices[self.params.target_species]
		))

		assert all_seqs.shape[0] == self.batchsize * 2, all_seqs.shape[0]

		# Everything should have the same len!
		assert binding_labels.shape == species_labels.shape, (binding_labels.shape, species_labels.shape)
		assert species_labels.shape == domain_labels.shape, (species_labels.shape, domain_labels.shape)

		# Here we assign the name "classifier" to the binding prediction task, and "discriminator"
		return {"sequence":all_seqs, "index":domain_labels}, {"classifier":binding_labels, "discriminator":species_labels}

	def get_binding_data(self):
		'''
			Using current filenames stored in self.posfile and self.negfile,
			load in all of the "binding" training data as coordinates only.
			Then, when it is time to fetch individual batches, a chunk of
			coordinates will be converted into one-hot encoded sequences
			ready for model input.

			All data here (the binding task) comes from the source domain ONLY.
			That is consistent with the UDA setting :)
		'''
		with open(self.posfile) as posf:
			pos_coords_tmp		= [line.split()[:3] for line in posf]
			self.pos_coords		= [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in pos_coords_tmp]
			self.pos_onehots	= self.source_converter(self.pos_coords)

		with open(self.negfile) as negf:
			neg_coords_tmp		= [line.split()[:3] for line in negf]
			self.neg_coords		= [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in neg_coords_tmp]
			self.neg_onehots	= self.source_converter(self.neg_coords)

	def get_species_data(self):
		'''
			Using current filenames stored in self.source_species_file
			and self.target_species_file. There are the "species-background"
			training data for both species. These examples are used to inform
			whatever domain-adaptive task we go with.

			All data here (the domain-adaptive task) comes from the BOTH domains.
		'''
		with open(self.source_species_file) as sourcef:
			source_coords_tmp	= [line.split()[:3] for line in sourcef]
			self.source_coords	= [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in source_coords_tmp]
			self.source_onehots	= self.source_converter(self.source_coords)

		with open(self.target_species_file) as targetf:
			target_coords_tmp	= [line.split()[:3] for line in targetf]
			self.target_coords	= [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in target_coords_tmp]
			self.target_onehots	= self.target_converter(self.target_coords)

	def on_epoch_end(self):
		print(f"\n\nFinished training on the set (binding, neg): {self.negfile}\n")
		print(f"\nFinished training on the set (source, background): {self.source_species_file}\n")
		print(f"\nFinished training on the set (target, background): {self.target_species_file}\n\n")

		# (1) Switch to next set of negative examples
		prev_epoch = self.current_epoch
		self.current_epoch = prev_epoch + 1

		# (2) Update files to pull data from, for unbound examples and species-background examples
		prev_negfile = self.negfile
		next_negfile = prev_negfile.replace(str(prev_epoch) + "E", str(self.current_epoch) + "E")
		self.negfile = next_negfile

		if self.total_epochs < self.current_epoch:
			return
		else:
			prev_sourcefile = self.source_species_file
			next_sourcefile = prev_sourcefile.replace(str(prev_epoch) + "E", str(self.current_epoch) + "E")
			self.source_species_file = next_sourcefile

			prev_targetfile = self.target_species_file
			next_targetfile = prev_targetfile.replace(str(prev_epoch) + "E", str(self.current_epoch) + "E")
			self.target_species_file = next_targetfile

			# (3) Load new data into memory for unbound examples & species-background examples		
			with open(self.negfile) as negf:
				neg_coords_tmp = [line.split()[:3] for line in negf]
				self.neg_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in neg_coords_tmp]
				self.neg_onehots = self.source_converter(self.neg_coords)

			with open(self.source_species_file) as sourcef:
				source_coords_tmp = [line.split()[:3] for line in sourcef]
				self.source_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in source_coords_tmp]
				self.source_onehots = self.source_converter(self.source_coords)

			with open(self.target_species_file) as targetf:
				target_coords_tmp = [line.split()[:3] for line in targetf]
				self.target_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in target_coords_tmp]
				self.target_onehots = self.target_converter(self.target_coords)

			# (4) Then shuffle positive examples
			self.pos_onehots = tf.random.shuffle(self.pos_onehots)

class ValGenerator(Sequence):
	'''
		This class implements an iterable data loader for validation on
		data from a given species (either souce/training or target. This
		class loads in coordinates from the val set examples file,
		and then when a batch is accessed, it converts that batch's worth
		of coordinates into one-hot encoded sequences.
		
		This generator does not return labels -- those must be loaded in
		separately if performance evaluation is the goal.
	'''
	def __init__(self, params, target_species=False, workers=1, use_multiprocessing=False, max_queue_size=10):
		super().__init__(workers=workers, use_multiprocessing=use_multiprocessing, max_queue_size=max_queue_size)
		self.params = params
		if target_species:
			self.valfile			= params.targetvalfile
			self.steps_per_epoch	= params.target_val_steps
			self.converter			= PyfaidxCoordsToVals(params.target_genome_file)
			self.the_species		= params.target_species
		else:
			self.valfile			= params.sourcevalfile
			self.steps_per_epoch	= params.source_val_steps
			self.converter			= PyfaidxCoordsToVals(params.source_genome_file)
			self.the_species		= params.source_species

		self.batchsize = params.valbatchsize
		self.get_data()

	def __len__(self):
		return self.steps_per_epoch

	def __getitem__(self, batch_index):
		onehots_batch = self.onehots[batch_index * self.batchsize : (batch_index + 1) * self.batchsize]
		assert onehots_batch.shape[0] > 0, onehots_batch.shape[0]

		# Auxillary variable representing the domain index (species) for each example
		domain_labels = np.ones(onehots_batch.shape[0],) * self.params.domain_indices[self.the_species]

		return {"sequence":onehots_batch, "index":domain_labels}

	def get_data(self):
		with open(self.valfile) as f:
			coords_tmp = [line.split()[:3] for line in f]
			self.coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]
			self.onehots = self.converter(self.coords)

class TestGenerator(Sequence):
	"""
    This generator retrieves all coordinates for windows in the test set
    and converts the sequences in those windows to one-hot encodings.
    Which species to retrieve test windows for is specified with
    the "test_species" argument. 
	"""
	def __init__(self, params, batchsize, test_species, workers=1, use_multiprocessing=False, max_queue_size=10):
		super().__init__(workers=workers, use_multiprocessing=use_multiprocessing, max_queue_size=max_queue_size)
		self.params			= params
		self.converter		= PyfaidxCoordsToVals(GENOMES[test_species])
		self.batchsize		= batchsize
		self.the_species	= test_species

		# Rely on methods for these datapoints
		self.testfile	= self.get_test_bed_file(test_species)
		self.get_steps(batchsize)
		self.get_data()

	def __len__(self):
		return self.test_steps

	def __getitem__(self, batch_index):
		onehots_batch = self.onehots[batch_index * self.batchsize : (batch_index + 1) * self.batchsize]
		assert onehots_batch.shape[0] > 0, onehots_batch.shape[0]

		# Auxillary variable representing the domain index (species) for each example
		domain_labels = np.ones(onehots_batch.shape[0],) * self.params.domain_indices[self.the_species]

		return {"sequence":onehots_batch, "index":domain_labels}

	def get_steps(self, batchsize):
		# Calculates the number of steps needed to get through
		# all batches of examples in the test dataset
		# (Keras predict_generator code needs to know this)
		with open(self.testfile) as f:
			lines_in_file = sum(1 for line in f)
		self.test_steps = lines_in_file // batchsize

	def get_test_bed_file(self, test_species):
		# This function returns the path to a BED-format file
		# containing the chromosome names, starts, and ends for
		# all examples to test the model with.

		# NOTE: binding labels will not be loaded in.
		# This file should contain the same examples for any TF.

		# It doesn't matter which TF we pick because we only need the coordinates!
		# They are shared in across all TFs, what differs is the labels attached 
		# to them.
		return(f"{ROOT}/data/{test_species}/CTCF/chr2.bed")

	def get_data(self):
		# Load all coordinates for the test data into memory
		with open(self.testfile) as f:
			coords_tmp = [line.rstrip().split()[:3] for line in f]

		assert [len(line_split) == 3 for line_split in coords_tmp]

		self.coords		= [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]
		self.onehots	= self.converter(self.coords)
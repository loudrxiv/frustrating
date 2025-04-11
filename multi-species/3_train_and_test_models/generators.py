'''
We implement several map-style datasets here for the purpose of creating
a base for the eventual dataloader we will use to evaluate our models.

Something we rely on right now is the seqdataloader package 
(https://github.com/kundajelab/seqdataloader), which uses a fasta index
to quickly generate one-hot encodings of sequences from a bed-formatted 
coordinate set corresponding to a region in the genome.
'''

import numpy as np
import random
import torch

from math import floor, ceil
from params import ROOT, GENOMES
from torch.utils.data import Dataset
from seqdataloader.batchproducers.coordbased.core import Coordinates
from seqdataloader.batchproducers.coordbased.coordstovals.fasta import PyfaidxCoordsToVals

#--- We need these for both single-species and multi-species models!

class DataGenerator(Dataset):
	def __init__(self, params, species, percent_to_batch):
		# Point to the files we need to fetch examples from!
		self.speciesfile	= params.speciestrainfiles[species]
		self.posfile		= params.bindingtrainposfiles[species]
		self.negfile		= params.bindingtrainnegfiles[species]

		# This operated on a specified species, so we generate a converter for that species
		self.converter		= PyfaidxCoordsToVals(params.genome_files[species])

		# Deal with domain indices
		self.domain_index	= params.domain_indices[species]

		# Now we can set the length of the dataset. We move 
		# forward with the knowledge that there will always be
		# less positive examples than negative examples or background
		# examples. So with the # of epochs in mind, we truncate the
		# background & negative examples accordingly! (Yes, we shuffled
		# all these files beforehand).
		self.current_epoch	= 0
		self.pos_len		= params.lens[species]['pos']
		
		# We also convert the number of coordinates, batch-wise, to speed up training
		if (percent_to_batch >= 0 and percent_to_batch <= 1):
			self.num_coords_to_do = int(floor(self.pos_len * percent_to_batch))
		else:
			raise ValueError("num_coords_to_do must be a percentage! (between 0 and 1).")
		
		# Formulate coordinates for all files
		self.get_data()

	def __len__(self):
		return self.pos_len

	def __getitem__(self, index):
		do_index = index % self.num_coords_to_do

		# Since any conversion (PyfaidxCoordsToVals, or even genompy) is slow,
		# we do it in batches that are specified at init. We read them from that
		# point on. We need to make sure that if the index is divisible by the number
		# of coordinates to do, we get the next batch of coordinates.
		if (do_index == 0):
			self.pos_ohe_batch		= self.converter(self.pos_coords_batch[index:index+self.num_coords_to_do])
			self.neg_ohe_batch		= self.converter(self.neg_coords_batch[index:index+self.num_coords_to_do])
			self.species_ohe_batch	= self.converter(self.species_coords_batch[index:index+self.num_coords_to_do])
			self.species_lab_batch	= self.species_labels_batch[index:index+self.num_coords_to_do]

		# Get positive example
		pos_ohe = self.pos_ohe_batch[do_index]

		# Get negative example
		neg_ohe = self.neg_ohe_batch[do_index]

		# Get a species example
		species_ohe = self.species_ohe_batch[do_index]
		species_lab = self.species_lab_batch[do_index]

		return {
			"sequence": torch.cat((
				torch.from_numpy(pos_ohe).reshape(1, 1000, 4),
				torch.from_numpy(neg_ohe).reshape(1, 1000, 4),
				torch.from_numpy(species_ohe).reshape(1, 1000, 4)
			), dim=0),
			"label": torch.from_numpy(np.array([1, 0, species_lab])),
			"index": torch.from_numpy(np.array([self.domain_index, self.domain_index, self.domain_index]))
		}

	def set_epoch(self, epoch):
		self.current_epoch = epoch

		# Reshuffle the positive examples for each epoch
		random.shuffle(self.pos_coords)

		# Get new index range (based on epoch)
		start_index	= self.pos_len * self.current_epoch
		end_index	= start_index + self.pos_len

		print("\nLooking at indices from {} to {}.\n".format(start_index, end_index))

		self.pos_coords_batch		= self.pos_coords
		self.neg_coords_batch		= self.neg_coords[start_index:end_index]
		self.species_coords_batch	= self.species_coords[start_index:end_index]
		self.species_labels_batch	= self.species_labs[start_index:end_index]

	def get_data(self):
		# Get positive example
		pos_coords_tmp	= [line.split()[:3] for line in open(self.posfile)]
		self.pos_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in pos_coords_tmp]

		# Get negative example (truncated)
		neg_coords_tmp	= [line.split()[:3] for line in open(self.negfile)]
		self.neg_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in neg_coords_tmp]

		# Get species example (truncated)
		species_coords_tmp	= [line.split() for line in open(self.speciesfile)]
		self.species_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in species_coords_tmp]
		self.species_labs 	= [int(coord[3]) for coord in species_coords_tmp]

#--- For target-on-target models!

class TrainGenerator_SingleSpecies(Dataset):
	def __init__(self, params, percent_to_batch):
		super().__init__()
		self.data	= DataGenerator(params=params, species=params.target_species, percent_to_batch=percent_to_batch)
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		data									= self.data[index]
		pos_ohe, neg_ohe, background_ohe		= data['sequence']
		pos_lab, neg_lab, background_lab		= data['label']
		pos_index, neg_index, background_index	= data['index']

		items = {
			"sequence": {
				"binding": {
					"positive": pos_ohe,
					"negative": neg_ohe
				},
				"background": background_ohe
			},
			"index": {
				"binding": {
					"positive": pos_index,
					"negative": neg_index
				},
				"background": background_index
			},
			"label": {
				"binding": {
					"positive": pos_lab,
					"negative": neg_lab
				},
				"background": background_lab
			}
		}

		return items
	
class ValGenerator_SingleSpecies(Dataset):
	def __init__(self, params, percent_to_batch):
		super().__init__()
		species					= params.target_species
		self.valfile			= params.valfiles[species]
		self.converter			= PyfaidxCoordsToVals(params.genome_files[species])
		self.domain_index		= params.domain_indices[species]

		# This is simpler! We just get the length of the validation file; however,
		# it is still LARGE. So we use 1,000,000 examples!
		self.val_len	= 1000000

		# We also convert the number of coordinates, batch-wise, to speed up training
		assert percent_to_batch >= 0.0 and percent_to_batch <= 1.0, "num_coords_to_do must be a percentage! (between 0 and 1)."

		self.num_coords_to_do = int(floor(self.val_len * percent_to_batch))

		# Formulate data for all files
		self.get_data()

	def __len__(self):
		return self.val_len

	def __getitem__(self, index):
		do_index = index % self.num_coords_to_do

		if (do_index == 0):
			self.val_ohe_batch	= self.converter(self.val_coords[index:index+self.num_coords_to_do])
			self.val_lab_batch	= self.val_labels[index:index+self.num_coords_to_do]

		# Get positive example
		val_ohe = self.val_ohe_batch[do_index]
		val_lab = self.val_lab_batch[do_index]

		items = {
			"sequence": torch.from_numpy(val_ohe).reshape(1, 1000, 4),
			"label": torch.from_numpy(np.array(val_lab))
		}

		return items

	def get_data(self):
		with open(self.valfile) as f:
			coords_tmp = [line.split() for line in f][:self.val_len]
			self.val_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]
			self.val_labels = [int(coord[3]) for coord in coords_tmp]

class TestGenerator_SingleSpecies(Dataset):
	def __init__(self, params, percent_to_batch):
		super().__init__()
		species 			= params.target_species
		self.testfile		= params.testfiles[species]
		self.converter		= PyfaidxCoordsToVals(params.genome_files[species])
		self.domain_index	= params.domain_indices[species]

		# This is simpler! We just get the length of the test file; however,
		# it is still LARGE. So we use 2,000,000 examples!
		self.test_len = 2000000

		# We also convert the number of coordinates, batch-wise, to speed up training
		assert percent_to_batch >= 0.0 and percent_to_batch <= 1.0, "num_coords_to_do must be a percentage! (between 0 and 1)."

		self.num_coords_to_do = int(floor(self.test_len * percent_to_batch))

		# Formulate data for all files
		self.get_data()

	def __len__(self):
		return self.test_len

	def __getitem__(self, index):
		do_index = index % self.num_coords_to_do

		if (do_index == 0):
			self.test_ohe_batch	= self.converter(self.test_coords[index:index+self.num_coords_to_do])
			self.test_lab_batch	= self.test_labels[index:index+self.num_coords_to_do]

		# Get positive example
		test_ohe = self.test_ohe_batch[do_index]
		test_lab = self.test_lab_batch[do_index]

		items = {
			"sequence": torch.from_numpy(test_ohe).reshape(1, 1000, 4),
			"label": torch.from_numpy(np.array(test_lab))
		}

		return items
		
	def get_data(self):
		with open(self.testfile) as f:
			coords_tmp = [line.split() for line in f][:self.test_len]
			self.test_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]
			self.test_labels = [int(coord[3]) for coord in coords_tmp]

#--- For multi-species models!

class TrainGenerator_MultiSpecies(Dataset):
	def __init__(self, params, percent_to_batch):
		super().__init__()
		# Put together datasets
		self.source_datasets	= [DataGenerator(params=params, species=s, percent_to_batch=percent_to_batch) for s in params.source_species]
		self.target_dataset		= DataGenerator(params=params, species=params.target_species, percent_to_batch=percent_to_batch)

		# Find the minimum length of all binding datasets
		self.min_len = min(min([len(ds) for ds in self.source_datasets]), len(self.target_dataset))

	def __len__(self):
		'''
		We consider the end of training to be when we reach the end of the shortest dataset.
		'''
		return self.min_len

	def __getitem__(self, index):
		# We want all datatypes from the source species but only the 
		# background sequences from the target species.
		source_samples = [ds[index] for ds in self.source_datasets]
		target_sample = self.target_dataset[index]

		# (1) seperate the source, sequence data into the different types
		pos_ohes		= [ss['sequence'][0].reshape(1, 1000, 4) for ss in source_samples]
		neg_ohes		= [ss['sequence'][1].reshape(1, 1000, 4) for ss in source_samples]
		species_ohes	= [ss['sequence'][2].reshape(1, 1000, 4) for ss in source_samples]
		species_ohes.append(target_sample['sequence'][2].reshape(1, 1000, 4))

		# (2) seperate the source, label data into the different types
		pos_labs		= np.array([ss['label'][0] for ss in source_samples])
		neg_labs		= np.array([ss['label'][1] for ss in source_samples])
		species_labs	= [ss['label'][2] for ss in source_samples]
		species_labs.append(target_sample['label'][2])

		# (3) seperate the source, index data into the different types
		pos_index		= np.array([ss['index'][0] for ss in source_samples])
		neg_index		= np.array([ss['index'][1] for ss in source_samples])
		species_index	= [ss['index'][2] for ss in source_samples]
		species_index.append(target_sample['index'][2])

		items = {
			"sequence": {
				"binding": {
					"positive": torch.cat(pos_ohes),
					"negative": torch.cat(neg_ohes)
				},
				"background": torch.cat(species_ohes)
			},
			"index": {
				"binding": {
					"positive": pos_index, 
					"negative": neg_index
				},
				"background": species_index
			},
			"label": {
				"binding": {
					"positive": pos_labs,
					"negative": neg_labs
				},
				"background": species_labs
			}
		}

		return items

class ValGenerator_MultiSpecies(Dataset):
	def __init__(self, params, percent_to_batch):
		super().__init__()
		self.valfiles		= params.valfiles
		self.genomefiles	= params.genome_files
		self.converters		= {species: PyfaidxCoordsToVals(path) for species, path in self.genomefiles.items()}

		# This is simpler! We just get the length of the validation file; however,
		# it is still LARGE. So we use 1,000,000 examples!
		self.val_len	= 1000000

		# We also convert the number of coordinates, batch-wise, to speed up training
		assert percent_to_batch >= 0.0 and percent_to_batch <= 1.0, "num_coords_to_do must be a percentage! (between 0 and 1)."

		self.num_coords_to_do = int(floor(self.val_len * percent_to_batch))
		
		# Formulate coordinates for all files
		self.data = {species: (self.get_data(file)) for species, file in self.valfiles.items()}

	def __len__(self):
		return self.val_len

	def __getitem__(self, index):
		do_index = index % self.num_coords_to_do

		# Since any conversion (PyfaidxCoordsToVals, or even genompy) is slow,
		# we do it in batches that are specified at init. We read them from that
		# point on. We need to make sure that if the index is divisible by the number
		# of coordinates to do, we get the next batch of coordinates.
		if (do_index == 0):
			self.ohe_batches	= {species: self.converters[species](self.data[species][0][index:index+self.num_coords_to_do]) for species in self.valfiles.keys()}
			self.lab_batches	= {species: self.data[species][1][index:index+self.num_coords_to_do] for species in self.valfiles.keys()}

		# Get examples
		ohes = {
			species: torch.from_numpy(np.array(
				self.ohe_batches[species][do_index]
			)).reshape(1, 1000, 4) for species in self.valfiles.keys()
		}

		labs = {
			species: torch.from_numpy(np.array(
				self.lab_batches[species][do_index]
			)) for species in self.valfiles.keys()
		}

		items = {
			"sequence": ohes, #torch.from_numpy(ohes).reshape(len(labs), 1000, 4),
			"label": labs #torch.from_numpy(labs)
		}

		return items
	
	def get_data(self, valfile):
		with open(valfile) as f:
			coords_tmp = [line.split() for line in f][:self.val_len]
			coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]
			labels = [int(coord[3]) for coord in coords_tmp]
		return coords, labels

class TestGenerator_MultiSpecies(Dataset):
	def __init__(self, params, percent_to_batch):
		super().__init__()
		self.testfiles		= params.testfiles
		self.genomefiles	= params.genome_files
		self.converters		= {species: PyfaidxCoordsToVals(path) for species, path in self.genomefiles.items()}

		# This is simpler! We just get the length of the test file; however,
		# it is still LARGE. So we use 2,000,000 examples!
		self.test_len = 2000000

		# We also convert the number of coordinates, batch-wise, to speed up training
		assert percent_to_batch >= 0.0 and percent_to_batch <= 1.0, "num_coords_to_do must be a percentage! (between 0 and 1)."

		self.num_coords_to_do = int(floor(self.test_len * percent_to_batch))

		# Formulate coordinates for all files
		self.data = {species: (self.get_data(file)) for species, file in self.testfiles.items()}

	def __len__(self):
		return self.test_len

	def __getitem__(self, index):
		do_index = index % self.num_coords_to_do

		# Since any conversion (PyfaidxCoordsToVals, or even genompy) is slow,
		# we do it in batches that are specified at init. We read them from that
		# point on. We need to make sure that if the index is divisible by the number
		# of coordinates to do, we get the next batch of coordinates.
		if (do_index == 0):
			self.ohe_batches	= {species: self.converters[species](self.data[species][0][index:index+self.num_coords_to_do]) for species in self.testfiles.keys()}
			self.lab_batches	= {species: self.data[species][1][index:index+self.num_coords_to_do] for species in self.testfiles.keys()}

		# Get examples
		ohes = {
			species: torch.from_numpy(np.array(
				self.ohe_batches[species][do_index]
			)).reshape(1, 1000, 4) for species in self.testfiles.keys()
		}

		labs = {
			species: torch.from_numpy(np.array(
				self.lab_batches[species][do_index]
			)) for species in self.testfiles.keys()
		}

		items = {
			"sequence": ohes, #torch.from_numpy(ohes).reshape(len(labs), 1000, 4),
			"label": labs #torch.from_numpy(labs)
		}

		return items
		
	def get_data(self, testfile):
		with open(testfile) as f:
			coords_tmp = [line.split() for line in f][:self.test_len]
			coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]
			labels = [int(coord[3]) for coord in coords_tmp]
		return coords, labels
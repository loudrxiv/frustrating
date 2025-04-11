from math import floor
from subprocess import check_output
from pprint import pprint
import numpy as np
from datetime import datetime
import os

# Declare species
SPECIES	= ["mm10", "hg38", "rheMac10", "canFam6", "rn7"]

# Declare transciption factors (we want to make quicker prototype sets too)
TFS		= ["CEBPA", "FOXA1", "HNF4A", "HNF6"]

# Delare root
ROOT	= "/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/MORALE/multi-species"

# We instantiate filenames for the pertinent files
VAL_FILENAME		= "val_shuf.bed"
TEST_FILENAME		= "test_shuf.bed"
TRAIN_FILENAME		= "train_shuf.bed"
TRAIN_POS_FILENAME	= "train_pos_shuf.bed"
TRAIN_NEG_FILENAME	= "train_neg_shuf.bed"

# Need to provide seqdataloader with locations of genome fasta files
# seqdataloader is expecting that there will be a fasta index in the
# same directory
GENOMES = {
    "mm10" : f"{ROOT}/raw_data/mm10/mm10.fa",
    "hg38" : f"{ROOT}/raw_data/hg38/hg38.fa",
    "rheMac10": f"{ROOT}/raw_data/rheMac10/rheMac10.fa",
    "canFam6": f"{ROOT}/raw_data/canFam6/canFam6.fa",
    "rn7": f"{ROOT}/raw_data/rn7/rn7.fa"
}

# Other constants for ease
DATA_ROOT			= f"{ROOT}/data"
MODEL_ROOT			= f"{ROOT}/models"
TIMESTAMP_FORMAT	= "%Y-%m-%d_%H-%M-%S"

class Params:
	'''
	The parameters we instantiate for use in our overall training scheme! This is used for
	all models: baseline, joint, and domain-adaptive models.
	'''
	def __init__(self, args, verbose=True):
		self.seqlen			= 1000		# the input sequence length that will be expected by the model
		self.convfilters	= 240		# number of filters in the convolutional layer
		self.filtersize		= 20		# the size of the convolutional filters
		self.strides		= 15		# the max-pooling layer's stride
		self.pool_size		= 15		# the max-pooling layer's pooling size
		self.lstmnodes		= 32		# "width" of the LSTM layer
		self.dl1nodes		= 1024		# neurons in the first dense layer (after LSTM)
		self.dl2nodes		= 512		# neurons in the second dense layer (before output)
		self.dropout		= 0.5		# fraction of neurons in the first dense layer to randomly dropout
		self.valbatchsize	= 10000
		self.testbatchsize	= self.valbatchsize
		self.epochs			= 15

		# (1) Parse the arguments received
		self.parse_args(args)
		
		# (2) Set the lengths of the training data, validation data, and test data
		self.set_lens()

		# (3) Set the validation, and test labels
		self.set_labels()

		if verbose:
			pprint(vars(self))

	def parse_args(self, args):
		timestamp = datetime.now().strftime(TIMESTAMP_FORMAT) # we use this for a variety of things

		# (1) Assess passed in arguments
		assert len(args) == 3, len(args)

		self.model_name		= args[0]

		self.tf				= args[1]
		assert self.tf in TFS, self.tf

		self.target_species	= args[2]
		assert self.target_species in SPECIES, self.target_species

		# (2) Set some things we need to keep track of
		self.source_species	= [species for species in SPECIES if species != self.target_species]
		self.genome_files	= {species: GENOMES[species] for species in SPECIES}
		roots				= {i: f"{DATA_ROOT}/{i}" for i in SPECIES}
		self.domain_indices = {species: i for i, (species, path) in enumerate(roots.items())}

		# (3) Set the paths for training, validation, and testing
		self.bindingtrainposfiles	= {species: f"{path}/{self.tf}/{TRAIN_POS_FILENAME}" for (species, path) in roots.items()}
		self.bindingtrainnegfiles	= {species: f"{path}/{self.tf}/{TRAIN_NEG_FILENAME}" for (species, path) in roots.items()}
		self.speciestrainfiles		= {species: f"{path}/{self.tf}/{TRAIN_FILENAME}" for (species, path) in roots.items()}
		self.valfiles				= {species: f"{path}/{self.tf}/{VAL_FILENAME}" for (species, path) in roots.items()}
		self.testfiles				= {species: f"{path}/{self.tf}/{TEST_FILENAME}" for (species, path) in roots.items()}

		# Setup model files and directories
		self.modeldir = f"{MODEL_ROOT}/{self.tf}/{self.target_species}_tested/{self.model_name}"
		os.makedirs(self.modeldir, exist_ok=True)

		self.modelfile		= f"{self.modeldir}/{timestamp}"

	def get_output_path(self): 
		return self.modelfile.split(".")[0] + ".probs.out"

	def set_lens(self):
		'''
		We calculate the total number of samples we could have access to for training.
		'''
		
		# Three components to training: 
		# 	1. the positive examples from the source species
		# 	2. the negative examples from the source species
		# 	3. random examples from all species

		#--- (1) Get Positive example from source species
		pos_commands 	= {species: ["wc", "-l", path] for species, path in self.bindingtrainposfiles.items()}
		pos_linecounts	= {species: int(check_output(command).strip().split()[0]) for species, command in pos_commands.items()}

		#-- (2) Get Negative examples from all species
		neg_commands	= {species: ["wc", "-l", path] for species, path in self.bindingtrainnegfiles.items()}
		neg_linecounts	= {species: int(check_output(command).strip().split()[0]) for species, command in neg_commands.items()}

		#--- (3) Get Random examples from all species
		species_commands	= {species: ["wc", "-l", path] for species, path in self.speciestrainfiles.items()}
		species_linecounts	= {species: int(check_output(command).strip().split()[0]) for species, command in species_commands.items()}

		#--- (4) Get validation lengths for each species
		validation_commands = {species: ["wc", "-l", path] for species, path in self.valfiles.items()}
		validation_linecounts = {species: int(check_output(command).strip().split()[0]) for species, command in validation_commands.items()}

		#--- (5) Get test lengths for each species
		test_commands = {species: ["wc", "-l", path] for species, path in self.testfiles.items()}
		test_linecounts = {species: int(check_output(command).strip().split()[0]) for species, command in test_commands.items()}

		self.lens 	= {
			species: {
				"pos": pos_linecounts[species],
				"neg": neg_linecounts[species],
				"species": species_linecounts[species],
				"validation": validation_linecounts[species],
				"test": test_linecounts[species]
			} for species in species_commands.keys()
		}

	def set_labels(self):
		'''
		We set the validation/test labels now for all species that we train and test on.
		Doing this now avoids having to repeatedly do it every epoch as part of
		model evaluation. The labels are read in as numpy arrays containing binary values.
		'''
		# Validation labels
		self.val_labels	= {species: np. array([int(line.split()[-1]) for line in open(path)]) for species, path in self.valfiles.items()}
		self.val_steps	= {species: int(floor(self.val_labels[species].shape[0] / self.valbatchsize)) for species, path in self.valfiles.items()}
		self.val_labels	= {species: self.val_labels[species][:self.val_steps[species] * self.valbatchsize] for species, path in self.valfiles.items()}

		# Test labels
		self.test_labels = {species: np.array([int(line.split()[-1]) for line in open(path)]) for species, path in self.testfiles.items()}
		self.test_steps	= {species: int(floor(self.test_labels[species].shape[0] / self.testbatchsize)) for species, path in self.testfiles.items()}
		self.test_labels = {species: self.test_labels[species][:self.test_steps[species] * self.testbatchsize] for species, path in self.testfiles.items()}
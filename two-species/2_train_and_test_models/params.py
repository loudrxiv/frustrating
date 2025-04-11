from __future__ import division # needs to be at the top of the file

import os
import numpy as np

from math import floor
from subprocess import check_output
from pprint import pprint
from datetime import datetime
from math import ceil, floor
from pprint import pprint

ROOT		= os.path.expanduser('~/Documents/research/dennis/domain_adaptation/MORALE/two-species')
DATA_ROOT	= f"{ROOT}/data"
SPECIES 	= ["mm10", "hg38"]
TFS			= ["CTCF", "CEBPA", "HNF4A", "RXRA"]

# Need to provide seqdataloader with locations of genome fasta files
# seqdataloader is expecting that there will be a fasta index in the same directory
GENOMES = {
	"mm10" : f"{ROOT}/raw_data/mm10/mm10_no_alt_analysis_set_ENCODE.fasta",
	"hg38" : f"{ROOT}/raw_data/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
}

# These files are created by the script 1_make_training_and_testing_data/1_runall_setup_model_data.sh
VAL_FILENAME 		= "chr1_random_1m.bed"
TEST_FILENAME		= "chr2.bed"
TRAIN_POS_FILENAME	= "chr3toY_pos_shuf.bed"
TRAIN_NEG_FILENAME	= "chr3toY_neg_shuf_runX_1E.bed"
SPECIES_FILENAME	= "chr3toY_shuf_runX_1E.bed"

# Where models will be saved during/after training
MODEL_ROOT			= f"{ROOT}/models"
TIMESTAMP_FORMAT	= "%Y-%m-%d_%H-%M-%S"

class Params:
	'''
		This class is a data structure that contains the hard-coded
		parameters and filepaths needed by the rest of the model-training
		code.

		This class is specific to models that don't domain-adapt;
		See DA_params.py for the domain-adaptive model parameters. That class
		does inherit most of the parameters from this class, but there are
		some additional specifics.
	'''
	def __init__(self, args, verbose=True):
		self.batchsize		= 400	# number of examples seen every batch during training
		self.umap_size		= 5000	# size of the UMAP layer
		self.seqlen			= 500	# the input sequence length that will be expected by the model
		self.convfilters	= 240	# number of filters in the convolutional layer
		self.filtersize		= 20	# the size of the convolutional filters
		self.strides		= 15	# the max-pooling layer's stride
		self.pool_size		= 15	# the max-pooling layer's pooling size
		self.lstmnodes		= 32	# "width" of the LSTM layer
		self.dl1nodes		= 1024	# neurons in the first dense layer (after LSTM)
		self.dl2nodes		= 512	# neurons in the second dense layer (before output)
		self.dropout		= 0.5	# fraction of neurons in the first dense layer to randomly dropout
		self.valbatchsize	= 10000
		self.testbatchsize	= 1024
		self.epochs			= 15
		self.verbose		= verbose

		self.parse_args(args)
		self.set_steps_per_epoch()

		if self.verbose:
			pprint(vars(self))

		self.set_val_labels()
		self.set_test_labels()

	def parse_args(self, args):
		''' This method parses the info passed in (TF, source species, and run #)
			and determines the filepaths for the data, genomes, and model save location.

			This method is expecting that the data for each tf and species
			is organized in a particular directory structure. See
			setup_directories_and_download_files.sh for this directory structure.

			This method is also expecting arguments input in a particular order.
			Should be: TF, source species (mm10 or hg38), run number
		'''
		assert len(args) >= 4, len(args) # the first item in argv is the script name
		self.model_name = args[0]
		
		self.tf = args[1]
		assert self.tf in TFS, self.tf

		self.source_species = args[2]

		# We always stick with this naming convention for the training data files
		TRAIN_POS_FILENAME = "chr3toY_pos_shuf.bed"
		TRAIN_NEG_FILENAME = "chr3toY_neg_shuf_runX_1E.bed"
            
		assert self.source_species in SPECIES, self.source_species
		self.run = int(args[3])

		source_root = f"{DATA_ROOT}/{self.source_species}/{self.tf}"

		# The target species is just the opposite species from the source species
		self.target_species = [species for species in SPECIES if species != self.source_species][0]
		target_root = f"{DATA_ROOT}/{self.target_species}/{self.tf}"

		self.bindingtrainposfile = f"{source_root}/{TRAIN_POS_FILENAME}"

		# The file of negative/unbound examples is specific to each run and epoch.
		# For now we're loading in the filename for the first epoch, but we need to
		# make the filename run-specific.
		self.bindingtrainnegfile = f"{source_root}/{TRAIN_NEG_FILENAME}"
		self.bindingtrainnegfile = self.bindingtrainnegfile.replace("runX", "run" + str(self.run))

		self.sourcevalfile = f"{source_root}/{VAL_FILENAME}"
		self.targetvalfile = f"{target_root}/{VAL_FILENAME}"

		self.sourcetestfile = f"{source_root}/{TEST_FILENAME}"
		self.targettestfile = f"{target_root}/{TEST_FILENAME}"

		timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)

		self.modeldir = f"{MODEL_ROOT}/{self.tf}/{self.source_species}_trained/{self.model_name}"

		os.makedirs(self.modeldir, exist_ok=True)
		self.modelfile = f"{self.modeldir}/{timestamp}_run{str(self.run)}"

		self.source_genome_file = GENOMES[self.source_species]
		self.target_genome_file = GENOMES[self.target_species]

		# For the domain-adaptive models, we also need to provide this information
		self.domain_indices = {species: i for i, (species, genome_file) in enumerate(GENOMES.items())}

		# These paths are unique to the domain-adaptive models.
		# In addition to binding-labeled training data, the DA models also
		# use "species-background" training data from both species, without binding labels.
		self.source_species_file = source_root + "/" + SPECIES_FILENAME.replace("runX", "run" + str(self.run))
		self.target_species_file = target_root + "/" + SPECIES_FILENAME.replace("runX", "run" + str(self.run))

	def get_output_path(self):
		return self.modelfile.split(".")[0] + ".probs.out"

	def set_steps_per_epoch(self):
		''' This method determines the number of batches for the model to load
		for a complete epoch. It is defined by the floor of the number of examples
		in the training data divided by the batchsize.

		Because we know that exactly half our training data is bound, the size of
		the bound examples file is half the number of examples seen in an epoch.
		So, the # of train steps is floor(bound_examples * 2 / batchsize).

		Note that we assume the training set is balanced (50% bound examples).
		'''
		command = ["wc", "-l", self.bindingtrainposfile]
		linecount = int(check_output(command).strip().split()[0])
		self.train_steps = int(floor((linecount * 2) / self.batchsize))

	def set_val_labels(self):
		''' This method reads in the labels for the validation data for both species.
			Doing this now avoids having to repeatedly do it every epoch as part of
			model evaluation.
			
			The labels are read in as numpy arrays containing binary values. Because
			the dataset is truncated to have length that is a multiple of the batch size
			when fed into the model, we need to similarly truncate these labels. If your
			batch size is a factor of your validation dataset size, this doesn't have any effect.
		'''
		with open(self.targetvalfile) as f:
			self.target_val_labels = np.array([int(line.split()[-1]) for line in f])
		self.target_val_steps = int(floor(self.target_val_labels.shape[0] / self.valbatchsize))
		self.target_val_labels = self.target_val_labels[:self.target_val_steps * self.valbatchsize]

		with open(self.sourcevalfile) as f:
			self.source_val_labels = np.array([int(line.split()[-1]) for line in f])
		self.source_val_steps = int(floor(self.source_val_labels.shape[0] / self.valbatchsize))
		self.source_val_labels = self.source_val_labels[:self.source_val_steps * self.valbatchsize]

	def set_test_labels(self):
		''' This method reads in the labels for the validation data for both species.
			Doing this now avoids having to repeatedly do it every epoch as part of
			model evaluation.
			
			The labels are read in as numpy arrays containing binary values. Because
			the dataset is truncated to have length that is a multiple of the batch size
			when fed into the model, we need to similarly truncate these labels. If your
			batch size is a factor of your validation dataset size, this doesn't have any effect.
		'''
		with open(self.targettestfile) as f:
			self.target_test_labels	= np.array([int(line.split()[-1]) for line in f])
		self.target_test_steps		= int(floor(self.target_test_labels.shape[0] / self.testbatchsize))
		self.target_test_labels		= self.target_test_labels[:self.target_test_steps * self.testbatchsize]

		with open(self.sourcetestfile) as f:
			self.source_test_labels	= np.array([int(line.split()[-1]) for line in f])
		self.source_test_steps		= int(floor(self.source_test_labels.shape[0] / self.testbatchsize))
		self.source_test_labels		= self.source_test_labels[:self.source_test_steps * self.testbatchsize]

	def get_reshape_size(self):
		'''
		DA models contain a reshape layer above the convolutional filters/pooling.
		This layer feeds into the gradient reversal layer and then a dense layer,
		which requires dimensions to be flattened.

		When the Reshape layer is initialized, it needs to know what input shape
		to expect, so this method calculates that.
		'''
		return int(ceil(self.seqlen / self.strides) * self.convfilters)
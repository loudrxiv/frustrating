''' 
	This script trains a binary classifier neural network on
	TF binding data.

	The model is a hybrid CNN-LSTM architecture
	implemented in the basic_model function below.

	The specific hyperparameters of the model, as well as paths
	to files containing the datasets used for training and validation,
	are stored in the Params data structure (see default_params.py).

	See the TrainGenerator class in generators.py for how the
	data loading is implemented. The generator relies on the package
	seqdataloader to transform bed coordinates of example training
	sequences into one-hot encodings.

	During training, the performance (auPRC) of the model is 
	evaluated each epoch on the validation sets of both species via
	the MetricsHistory callback class. The results of this evaluation 
	are printed to stdout (if running run_training.sh, they will be
	written to the log file). The per-epoch performance on the 
	source/training species is used as part of a simple model selection
	protocol: the model with the best-so-far source species validation
	set performance is saved for downstream analysis.

	Separately, the ModelSaveCallback class saves the model each epoch.

	The model trains using the ADAM optimizer and binary cross-entropy loss.
'''

#==========================================================================================================

ROOT = "/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/RECOMB/two-species/tensorflow"

#==========================================================================================================

import sys
sys.path.append(f"{ROOT}")
sys.path.append(f"{ROOT}/cross-species-domain-adaptation/2_train_and_test_models")

import keras
from keras.layers import Dense, Dropout, Activation, LSTM, Input, Conv1D, MaxPooling1D
from keras.models import Model

from params import Params
from generators import TrainGenerator_SingleTask
from callbacks import MetricsHistory

#==========================================================================================================

def basic_model(params):

	# Inputs
	seq_input	= Input(shape = (params.seqlen, 4, ), name = 'sequence')
	index		= Input(shape = (1,), name = 'index') # we don't use this here

	# Basic, CNN model
	seq = Conv1D(params.convfilters, params.filtersize, padding = "same")(seq_input)
	seq = Activation("relu")(seq)
	seq = MaxPooling1D(padding = "same", strides = params.strides, pool_size = params.pool_size)(seq)
	seq = LSTM(params.lstmnodes)(seq)
	seq = Dense(params.dl1nodes, activation = "relu")(seq)
	seq = Dropout(params.dropout)(seq)
	seq = Dense(params.dl2nodes, activation = "sigmoid")(seq)
	seq = Dense(1)(seq)

	# Label
	result = Activation("sigmoid")(seq)

	return Model(inputs = [seq_input, index], outputs = result)

#==========================================================================================================

if __name__ == "__main__":

	# Manually set the parameters here.
	_, tf, source_species, fold = sys.argv

	# (1) Instantiate what we need for training and validation
	params				= Params(["BM", tf, source_species, fold])
	metrics_callback	= MetricsHistory(params, save=True)
	model				= basic_model(params)
	generator			= TrainGenerator_SingleTask(params)

	# (2) Compile the model
	model.compile(
		loss=keras.losses.BinaryCrossentropy(),
		optimizer=keras.optimizers.Adam(learning_rate=1e-3),
		metrics=[
			keras.metrics.AUC(name="auc")
		]
	)
	
	# Print a summary of the model
	print(model.summary())

	# (3) Finally, train the model; the callback will save the weights of the best model
	hist = model.fit(
		x=generator,
		epochs=params.epochs,
		steps_per_epoch=params.train_steps,
		callbacks=[
			metrics_callback
		]
	)
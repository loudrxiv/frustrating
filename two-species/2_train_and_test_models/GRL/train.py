''' 
	This script trains a binary classifier neural network on
	TF binding data.

	The model is a hybrid CNN-LSTM architecture augmented with a 
	second "head" and a gradient reversal layer, implemented in the
	DA_model function below. The second head is tasked with trying to
	discriminate between sequences from the source and target species,
	and trains on mutually exclusive data from the normal binding task head.

	The specific hyperparameters of the model, as well as paths
	to files containing the datasets used for training and validation,
	are stored in the DA_Params data structure (see default_params.py and
	DA_params.py).

	See the DATrainGenerator class in generators.py for how the
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

	The model trains using the ADAM optimizer and binary cross-entropy
	loss with one augmentation -- in order to train both the TF binding and
	species-discriminating "heads" within the same batch, when the two heads
	train on mutually exclusive datasets, the loss must be masked for one of
	the two heads for each example. Implemented in the custom_loss function
	below is standard BCE loss with that mask included. See the code for the
	DATrainGenerator class for details on how the masked labels are created.

	By default, the losses of the TF binding and species-discriminating heads
	are equally weighted because the same amount of examples are used each batch
	for each task and because the loss_weight parameter used below is 1. But
	this parameter can be changed to upweight or downweight one task relative
	to the other.
'''

#==========================================================================================================

ROOT = "/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/RECOMB/two-species/tensorflow"

#==========================================================================================================

import sys
sys.path.append(f"{ROOT}")
sys.path.append(f"{ROOT}/cross-species-domain-adaptation/2_train_and_test_models")

import tensorflow
import keras
from keras.layers import Dense, Reshape, Activation, LSTM, Input, Conv1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import LambdaCallback

from params import Params
from generators import TrainGenerator_MultiTask
from callbacks import MetricsHistory
from flipGradientTF import GradientReversal

#==========================================================================================================

def Lbce(y_true, y_pred):
	# The model will be trained using this loss function, which is identical to normal BCE
	# except when the label for an example is -1, that example is masked out for that task.
	# This allows for examples to only impact loss backpropagation for one of the two tasks.
	y_pred = tensorflow.boolean_mask(y_pred, tensorflow.not_equal(y_true, -1))
	y_true = tensorflow.boolean_mask(y_true, tensorflow.not_equal(y_true, -1))
	return keras.losses.BinaryCrossentropy()(y_true, y_pred)

def GRL(params, lambda_):
	# Here we specify the architecture of the domain-adaptive model.

	# Inputs
	seq_input	= Input(shape = (params.seqlen, 4, ), name = 'sequence')
	index		= Input(shape = (1,), name = 'index') # we don't use this here

	# Shared convolutional layer
	seq = Conv1D(params.convfilters, params.filtersize, padding="same")(seq_input)
	seq = Activation('relu')(seq)
	seq = MaxPooling1D(padding = "same", strides = params.strides, pool_size = params.pool_size)(seq)

	# Classifier technically includes the LSTM layer (from Cochran et al.)
	classifier		= LSTM(params.lstmnodes)(seq)
	classifier		= Dense(params.dl1nodes)(classifier)
	classifier		= Activation('relu')(classifier)
	classifier		= Dense(params.dl2nodes, activation = 'sigmoid')(classifier)
	class_result	= Dense(1, activation = 'sigmoid', name = "classifier")(classifier)

	# Discriminator acts upon 
	discriminator	= Reshape((params.get_reshape_size(), ))(seq)
	discriminator	= GradientReversal(lambda_)(discriminator)
	discriminator	= Dense(params.dl1nodes)(discriminator)
	discriminator	= Activation('relu')(discriminator)
	discriminator	= Dense(params.dl2nodes, activation = 'sigmoid')(discriminator)
	disc_result		= Dense(1, activation = 'sigmoid', name = "discriminator")(discriminator)

	model = Model(inputs = [seq_input, index], outputs = [class_result, disc_result])
	return model

#==========================================================================================================

if __name__ == "__main__":

	# Manually set the parameters here.
	_, tf, source_species, fold, lambda_ = sys.argv

	# (1) Instantiate what we need for training and validation
	params				= Params(["GRL", tf, source_species, fold])
	metrics_callback	= MetricsHistory(params, save=True)
	model				= GRL(params, lambda_)
	generator			= TrainGenerator_MultiTask(params)

	model.compile(
		loss={"classifier":Lbce, "discriminator":Lbce},			# for each of our two tasks!
		loss_weights={"classifier": 1.0, "discriminator": 1.0},
		optimizer=keras.optimizers.Adam(learning_rate=1e-3),
		metrics={"classifier":keras.metrics.AUC(name="cls_auc"), "discriminator":keras.metrics.AUC(name="disc_auc")}
	)

	print(model.summary())

	hist = model.fit(
		x=generator,
		epochs=params.epochs,
		steps_per_epoch=params.train_steps,
		callbacks=[
			metrics_callback
		]
	)

#==========================================================================================================
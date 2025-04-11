#==========================================================================================================

ROOT = "/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/RECOMB/two-species/tensorflow"

#==========================================================================================================

import os

import sys
sys.path.append(f"{ROOT}")
sys.path.append(f"{ROOT}/cross-species-domain-adaptation/2_train_and_test_models")

import numpy as np

import tensorflow
import keras
from keras.layers import Dense, Dropout, Activation, LSTM, concatenate, Input, Conv1D, MaxPooling1D, Flatten, Reshape
from keras.models import Model

from params import Params
from generators import TestGenerator
from flipGradientTF import GradientReversal

#==========================================================================================================

def Lbce(y_true, y_pred):
	# The model will be trained using this loss function, which is identical to normal BCE
	# except when the label for an example is -1, that example is masked out for that task.
	# This allows for examples to only impact loss backpropagation for one of the two tasks.
	y_pred = tensorflow.boolean_mask(y_pred, tensorflow.not_equal(y_true, -1))
	y_true = tensorflow.boolean_mask(y_true, tensorflow.not_equal(y_true, -1))
	return keras.losses.BinaryCrossentropy()(y_true, y_pred)

def GRL(params, lambda_=1.0):
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

def get_model_file(tf, source_species, run):
    # This function returns the filepath where the model for a given
    # TF, training species, and run is saved.
    # By default, the file for the best model across all training epochs
    # is returned, you can change model_type to select the last model instead.
    # This function specifically looks for the most recent model file,
    # if there are multiple for the same run-TF-species combo.
    try:
        run_int = int(run)
    except:
        print("Error: You need to pass in a run number that can be cast to int.")

    model_file_prefix = f"{ROOT}/models/{tf}/{source_species}_trained/GRL"
    model_file_suffix = f"_run{str(run)}.weights.h5"
    
    # Get all files that match the prefix and suffix
    files = [f for f in os.listdir(model_file_prefix) if f.endswith(model_file_suffix)]
    
    # sort files and return the one that is most recent
    latest_file = max([f"{model_file_prefix}/{f}" for f in files], key=os.path.getctime)
    return latest_file

def load_keras_model(tf, source_species, run, lambda_, model_file):
    params  = Params(["GRL", tf, source_species, run], verbose=False)
    model   = GRL(params, lambda_)
    model.load_weights(model_file)
    return model

def get_models_all_runs(tf, source_species, lambda_, runs=5):
    # load in models for all runs, for a given TF and training species
    # returns a list of Keras model objects
    models = []
    for run in range(runs):
        model_file = get_model_file(tf=tf, source_species=source_species, run=run+1)
        models.append(load_keras_model(tf=tf, source_species=source_species, run=run, lambda_=lambda_, model_file=model_file))
    return models

def get_preds_file(tf, source_species, domain):
    preds_root = f"{ROOT}/output"
    os.makedirs(preds_root, exist_ok=True)
    return f"{preds_root}/GRL_tf-{tf}_trained-{source_species}_tested-{domain}.preds"

def get_labels_file(tf, source_species, domain):
    preds_root = f"{ROOT}/output"
    os.makedirs(preds_root, exist_ok=True)
    return f"{preds_root}/GRL_tf-{tf}_trained-{source_species}_tested-{domain}.labels"

def get_preds_batched_fast(model, test_generator):
    # Make predictions for all test data using a specified model.
    # Batch_size can be as big as your compute can handle.
    # Use test_species = "mm10" to test on mouse data instead of human data.
    preds  = model.predict(test_generator)
    return np.squeeze(preds)

#==========================================================================================================

if __name__ == "__main__":
    _, tf, source_species, domain, lambda_ = sys.argv # the script name is the first arg

    # Generate params to grab labels for the model
    params = Params(["GRL", tf, source_species, 1]) # doesn't matter which run we use

    # Get labels we need based on the domain
    if domain == params.source_species:
        print(f"\nEvaluating on the source ({params.source_species}) domain.\n")
        labels = np.array(params.source_test_labels)
    else:
        print(f"\nEvaluating on the target ({params.target_species}) domain.\n")
        labels = np.array(params.target_test_labels)

    # Load all the models from our 5-fold cross-validation
    models = get_models_all_runs(tf=tf, source_species=source_species, lambda_=lambda_)

    # Generate predictions for all 5 independent model runs on domain data
    test_generator  = TestGenerator(params, batchsize=params.testbatchsize, test_species=domain)

    print("\nGenerating predictions...\n")
    
    all_model_preds = np.array(
        [get_preds_batched_fast(model=model, test_generator=test_generator) for model in models]
    )

    # The domain-adaptive model outputs two predictions per model, so we need to only
    # keep the classifier predictions.
    all_model_preds = all_model_preds[:, 0, :]

    assert len(all_model_preds.shape) == 2, all_model_preds.shape

    # Create file to save predictions and
    preds_file  = get_preds_file(tf=tf, source_species=source_species, domain=domain)
    labels_file = get_labels_file(tf=tf, source_species=source_species, domain=domain)

    # Save it
    np.save(preds_file, all_model_preds.T)
    np.save(labels_file, labels)
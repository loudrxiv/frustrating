#==========================================================================================================

ROOT = "/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/RECOMB/two-species/tensorflow"

#==========================================================================================================

import os

import sys
sys.path.append(f"{ROOT}")
sys.path.append(f"{ROOT}/cross-species-domain-adaptation/2_train_and_test_models")

import numpy as np

import keras
import tensorflow
import numpy as np

from params import Params
from generators import TestGenerator
from keras.layers import Dense, Dropout, Activation, LSTM, concatenate, Input, Conv1D, MaxPooling1D, Flatten, Reshape
from keras.models import Model

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

    model_file_prefix = f"{ROOT}/models/{tf}/{source_species}_trained/BM"
    model_file_suffix = f"_run{str(run)}.weights.h5"
    
    # Get all files that match the prefix and suffix
    files = [f for f in os.listdir(model_file_prefix) if f.endswith(model_file_suffix)]
    
    # sort files and return the one that is most recent
    latest_file = max([f"{model_file_prefix}/{f}" for f in files], key=os.path.getctime)
    return latest_file

def load_keras_model(tf, source_species, run, model_file):
    params  = Params(["BM", tf, source_species, run], verbose=False)
    model   = basic_model(params)
    model.load_weights(model_file)
    return model

def get_models_all_runs(tf, source_species, runs=5):
    # load in models for all runs, for a given TF and training species
    # returns a list of Keras model objects
    models = []
    for run in range(runs):
        model_file = get_model_file(tf=tf, source_species=source_species, run=run+1)
        models.append(load_keras_model(tf=tf, source_species=source_species, run=run, model_file=model_file))
    return models

def get_preds_file(tf, source_species, domain):
    preds_root = f"{ROOT}/output"
    os.makedirs(preds_root, exist_ok=True)
    return f"{preds_root}/BM_tf-{tf}_trained-{source_species}_tested-{domain}.preds"

def get_labels_file(tf, source_species, domain):
    preds_root = f"{ROOT}/output"
    os.makedirs(preds_root, exist_ok=True)
    return f"{preds_root}/BM_tf-{tf}_trained-{source_species}_tested-{domain}.labels"

def get_preds_batched_fast(model, test_generator):
    # Make predictions for all test data using a specified model.
    # Batch_size can be as big as your compute can handle.
    # Use test_species = "mm10" to test on mouse data instead of human data.
    preds  = model.predict(test_generator)
    return np.squeeze(preds)

#==========================================================================================================

if __name__ == "__main__":
    _, tf, source_species, domain = sys.argv # the script name is the first arg

    # Generate params to grab labels for the model
    params = Params(["BM", tf, source_species, 1]) # doesn't matter which run we use

    # Get labels we need based on the domain
    if domain == params.source_species:
        print(f"\nEvaluating on the source ({params.source_species}) domain.\n")
        labels = np.array(params.source_test_labels)
    else:
        print(f"\nEvaluating on the source ({params.source_species}) domain.\n")
        labels = np.array(params.target_test_labels)

    # Load all the models from our 5-fold cross-validation
    models = get_models_all_runs(tf=tf, source_species=source_species)

    # Generate predictions for all 5 independent model runs on domain data
    test_generator  = TestGenerator(params, batchsize=params.testbatchsize, test_species=domain)

    print("\nGenerating predictions...\n")
    
    all_model_preds = np.array(
        [get_preds_batched_fast(model=model, test_generator=test_generator) for model in models]
    )

    assert len(all_model_preds.shape) == 2, all_model_preds.shape

    # Create file to save predictions and
    preds_file  = get_preds_file(tf=tf, source_species=source_species, domain=domain)
    labels_file = get_labels_file(tf=tf, source_species=source_species, domain=domain)

    # Save it
    np.save(preds_file, all_model_preds.T)
    np.save(labels_file, labels)
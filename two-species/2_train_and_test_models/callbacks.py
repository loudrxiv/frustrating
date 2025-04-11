''' 
	This file contains classes that inherit from Keras's Callback class.
	These classes contain methods like on_epoch_end and on_epoch_begin
	that Keras automatically triggers at the appropriate times during
	model training.
'''

import numpy as np
import keras

from generators import ValGenerator
from keras.callbacks import Callback
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, log_loss

class MetricsHistory(Callback):
	''' This callback evaluates the performance of the model on the
		validation datasets for both the source/training and target
		species at the end of each epoch. It generates predictions,
		calculates the auPRC and other metrics, prints them to stdout,
		and then checks the source-species auPRC to see if it is the
		best-so-far over training. If so, it saves the model. This
		best-overall model will be used for most downstream analyses.
        
		This callback uses labels for the validation set examples that
		have been read in by the Params or DA_Params class (called
		self.params/parameters here).
	'''

	def __init__(self, parameters, save):
		self.auprcs = []

		# This variable is not called self.params because the Callback class
		# has some self.params already, and so this attribute will be re-written
		# unless we call it something else.
		self.parameters			= parameters
		self.save				= save
		
		# We define both generators at the start to avoid having to re-create then
		self.source_generator	= ValGenerator(parameters, target_species=False)
		self.target_generator	= ValGenerator(parameters, target_species=True)

	def on_train_begin(self, logs={}):
		params = self.parameters

		print("Validating on target species...")
		params.target_val_probs = self.model.predict(self.target_generator)
		target_auprc = self.print_val_metrics(params, target_data=True)

		print("Validating on source species...")
		params.source_val_probs = self.model.predict(self.source_generator)
		source_auprc = self.print_val_metrics(params, target_data=False)

		current_auprcs = self.auprcs
		current_auprcs.append(source_auprc)
		self.auprcs = current_auprcs

	def on_epoch_end(self, batch, logs={}):
		params = self.parameters
		
		print("Validating on target species...")
		params.target_val_probs = self.model.predict(self.target_generator)
		target_auprc = self.print_val_metrics(params, target_data=True)

		print("Validating on source species...")
		params.source_val_probs = self.model.predict(self.source_generator)
		source_auprc = self.print_val_metrics(params, target_data=False)
		
		current_auprcs = self.auprcs
		if len(current_auprcs) == 0 or source_auprc > max(current_auprcs):
			print("Best model so far! (source species) validation auPRC = ", source_auprc)
			if self.save:
				print("\n(Saving these weights...)\n")

				# Naming suffix cannot change here; this is because of how TensorFlow :eyeroll: work
				# https://github.com/keras-team/keras/blob/v3.3.3/keras/src/callbacks/model_checkpoint.py#L14-L416
				self.model.save_weights(params.modelfile + ".weights.h5")
		current_auprcs.append(source_auprc)
		self.auprcs = current_auprcs
		
	def print_val_metrics(self, params, epoch_end = True, target_data = True):
		if target_data:
			print("\n==== Target Species Validation ====")
			labels = np.array(params.target_val_labels)
			probs = np.array(params.target_val_probs)
		else:
			print("\n==== Source Species Validation ====")
			labels = np.array(params.source_val_labels)
			probs = np.array(params.source_val_probs)

		# If the model is the domain-adaptive architecture,
		# it will return 2 sets of predictions -- one for the binding
		# task and one for the species-discrimination task. We only
		# care out the binding predictions, so we'll toss out the others.
		if probs.shape[0] == 2:
			probs = probs[0]  # use only binding classifier preds
		probs = probs.squeeze()
		assert labels.shape == probs.shape, (labels.shape, probs.shape)

		auc = roc_auc_score(labels, probs)
		print("AUC:\t", auc)
		
		auPRC = average_precision_score(labels, probs)
		print("auPRC:\t", auPRC)
		
		loss = log_loss(labels, probs)  # this is binary cross-entropy
		print("Loss:\t", loss)

		self.print_confusion_matrix(labels, probs)

		return auPRC
    
	def print_confusion_matrix(self, y, probs, threshold = 0.5):
		''' This method prints a confusion matrix to stdout, as well as
			the precision and recall when the predicted probabilities
			output by the model are threshold-ed at 0.5. Because this
			threshold is somewhat arbitrary, other metrics of performance
			such as the auPRC of the loss are more informative single
			quantities, but the confusion matrix can be useful to look at
			to show relative false positive and false negative rates.
		'''
		npthresh = np.vectorize(lambda t: 1 if t >= threshold else 0)
		pred = npthresh(probs)
		conf_matrix = confusion_matrix(y, pred)

		print("Confusion Matrix at t = 0.5:\n", conf_matrix)

		# precision = TP / (TP + FP)
		print("Precision at t = 0.5:\t", conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1]))
		
		# recall = TP / (TP + FN)
		print("Recall at t = 0.5:\t", conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0]), "\n")
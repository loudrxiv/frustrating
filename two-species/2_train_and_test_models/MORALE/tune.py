#==========================================================================================================

ROOT = "/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/RECOMB/two-species/tensorflow"

#==========================================================================================================

import sys
sys.path.append(f"{ROOT}")
sys.path.append(f"{ROOT}/cross-species-domain-adaptation/2_train_and_test_models")

import tensorflow 
import numpy as np

import keras
from keras.layers import Dense, Dropout, Activation, LSTM, Input, Conv1D, MaxPooling1D, Reshape
from keras.models import Model

from math import ceil, floor
from params import Params
from generators import TrainGenerator_Adaptive
from callbacks import MetricsHistory

#==========================================================================================================

class MomentAlignmentLayer(keras.layers.Layer):
    """
    Layer that computes the moment alignment loss for the model.

    We define our operation for calculating and aligning the moments. The only
    thing of NOTE is that I am not sure, in the two-species case, how to handle
    if we need to align m->h and h->m.
    """
    def __init__(self, params, _match_mean=False, _lambda=1.0):
        super().__init__()
        self.domain_indices         = params.domain_indices.values()
        self._match_mean            = _match_mean
        self._lambda                = _lambda
        self.alignment              = keras.metrics.Mean(name="alignment")

    def call(self, inputs):
        assert len(inputs) == 2, "The layer must be called on a list of two inputs: the embedding and the index."

        embedding, index    = inputs

        start_dim           = tensorflow.shape(embedding)[0] // 2

        # Parts of the embedding we operate on
        background_embedding    = embedding[start_dim:, :]
        background_index        = index[start_dim:,]

        moment_alignment_loss = 0.0
        for i in self.domain_indices:
            for j in self.domain_indices:

                if i == j: # same domain
                    continue

                # Get the mask
                mask_i = background_index == int(i)
                mask_j = background_index == int(j)

                # Define the source and target features
                source_features = background_embedding[mask_i]
                target_features = background_embedding[mask_j]

                curr_loss = self.MomentAlignment(Xs=source_features, Xt=target_features, _match_mean=self._match_mean, _lambda=self._lambda)
                moment_alignment_loss += curr_loss
        self.add_loss(moment_alignment_loss)
        self.alignment.update_state(moment_alignment_loss)
                
        return inputs

    def MomentAlignment(self, Xs, Xt, _match_mean=False, _lambda=1.0):
        """
        https://github.com/antoinedemathelin/adapt/blob/master/adapt/feature_based/_deepcoral.py
        """
        #EPS = np.finfo(np.float32).eps
        EPS = tensorflow.experimental.numpy.finfo(tensorflow.experimental.numpy.float32).eps

        batch_size = tensorflow.cast(tensorflow.shape(Xs)[0], Xs.dtype)

        factor_1 = 1. / (batch_size - 1. + EPS)
        factor_2 = 1. / batch_size

        sum_src     = tensorflow.reduce_sum(Xs, axis=0)
        sum_src_row = tensorflow.reshape(sum_src, (1, -1))
        sum_src_col = tensorflow.reshape(sum_src, (-1, 1))

        cov_src = factor_1 * (
            tensorflow.matmul(tensorflow.transpose(Xs), Xs) -
            factor_2 * tensorflow.matmul(sum_src_col, sum_src_row)
        )

        sum_tgt     = tensorflow.reduce_sum(Xt, axis=0)
        sum_tgt_row = tensorflow.reshape(sum_tgt, (1, -1))
        sum_tgt_col = tensorflow.reshape(sum_tgt, (-1, 1))

        cov_tgt = factor_1 * (
            tensorflow.matmul(tensorflow.transpose(Xt), Xt) -
            factor_2 * tensorflow.matmul(sum_tgt_col, sum_tgt_row)
        )

        mean_src = tensorflow.reduce_mean(Xs, 0)
        mean_tgt = tensorflow.reduce_mean(Xt, 0)

        disc_loss_cov   = 0.25 * tensorflow.square(cov_src - cov_tgt)
        disc_loss_mean  = tensorflow.square(mean_src - mean_tgt)

        disc_loss_cov   = tensorflow.reduce_mean(disc_loss_cov)
        disc_loss_mean  = tensorflow.reduce_mean(disc_loss_mean)
        disc_loss       = _lambda * (disc_loss_cov + _match_mean * disc_loss_mean)

        return disc_loss
    
    @property
    def metrics(self):
        return [self.alignment]

def Lbce(y_true, y_pred):
	# The model will be trained using this loss function, which is identical to normal BCE
	# except when the label for an example is -1, that example is masked out for that task.
	# This allows for examples to only impact loss backpropagation for one of the two tasks.
	y_pred = tensorflow.boolean_mask(y_pred, tensorflow.not_equal(y_true, -1))
	y_true = tensorflow.boolean_mask(y_true, tensorflow.not_equal(y_true, -1))
	return keras.losses.BinaryCrossentropy()(y_true, y_pred)

def MORALE(params, _match_mean=False, _lambda=1.0):
    # Inputs
    seq_input	= Input(shape = (params.seqlen, 4, ), batch_size=params.batchsize *2, name='sequence')
    index		= Input(shape = (), batch_size=params.batchsize*2, name='index')

    # To the embedding of the sequence
    seq         = Conv1D(params.convfilters, params.filtersize, padding = "same")(seq_input)
    seq         = Activation("relu")(seq)
    seq         = MaxPooling1D(padding = "same", strides = params.strides, pool_size = params.pool_size)(seq)
    embedding   = LSTM(params.lstmnodes)(seq)

    # Apply moment alignment framework
    aligned_embedding, _ = MomentAlignmentLayer(params, _match_mean=_match_mean, _lambda=_lambda)([embedding, index])

    # Continue with the model, specifying the architecture for binding label prediction
    seq		= Dense(params.dl1nodes, activation = "relu")(aligned_embedding)
    seq		= Dropout(params.dropout)(seq)
    seq		= Dense(params.dl2nodes, activation = "sigmoid")(seq)
    seq		= Dense(1)(seq)
    result	= Activation("sigmoid")(seq)

    model = Model(inputs = [seq_input, index], outputs = result, name = "MORALE")

    return model

#==========================================================================================================

if __name__ == "__main__":

	# (0) Manually set the parameters here.
	_, tf, source_species, fold, match, lambda_ = sys.argv

	# (1) Set the parameters, callbacks, and generators for the model.
	params				= Params(["MORALE", tf, source_species, fold])
	metrics_callback    = MetricsHistory(params, save=False)
	generator           = TrainGenerator_Adaptive(params)
	model				= MORALE(params, _match_mean=bool(match), _lambda=float(lambda_))

	print(model.summary())

	# (2) Compile and train the model.
	model.compile(
		loss=Lbce,
		optimizer=keras.optimizers.Adam(learning_rate=1e-3),
		metrics=[
			keras.metrics.AUC(name="auc"),
			Lbce
		]
	)

	# (3) Fit the model.
	hist = model.fit(
		x=generator,
		epochs=params.epochs,
		steps_per_epoch=params.train_steps,
		callbacks=[
			metrics_callback
		]
	)
     
#==========================================================================================================
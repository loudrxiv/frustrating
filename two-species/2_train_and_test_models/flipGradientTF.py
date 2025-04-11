''' This file is modified from Miche Tonu's Gradient Reversal Layer code at:
    https://github.com/michetonu/gradient_reversal_keras_tf
    
    The gradient reversal layer implementation requires implementing
    what to do on the forward pass and the backward pass through the layer.
    On the forward pass, the GRL just outputs its input; on the backward
    pass, it outputs the gradient passing in times negative lambda, where
    lambda is a hyperparameter (in this workflow, set to 1).
'''


import tensorflow as tf
import keras.backend as K

from tensorflow.keras.layers import Layer

class GradientReversal(Layer):
    """
    This implementation is still take off the original inspiration! It still 
    flips the sign of gradient during training. based on
    https://github.com/michetonu/gradient_reversal_keras_tf. However, DA performance
    could not be achieved with the across newer tensorflow versions. So we opted to adapt 
    this implementation: https://gist.github.com/oO0oO0oO0o0o00/74dbcb352164348e5268203fdf95a04b
    where it was ported to tf 2.x
    """
    def __init__(self, λ=1, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.λ = λ

    @staticmethod
    @tf.custom_gradient
    def reverse_gradient(x, λ):
        # @tf.custom_gradient suggested by Hoa's comment at
        # https://stackoverflow.com/questions/60234725/how-to-use-gradient-override-map-with-tf-gradienttape-in-tf2-0
        return tf.identity(x), lambda dy: (-dy, None)

    def call(self, x):
        return self.reverse_gradient(x, self.λ)

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super(GradientReversal, self).get_config() | {'λ': self.λ}
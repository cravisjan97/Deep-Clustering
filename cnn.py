import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
class binarize(theano.Op):
    """
    This creates an Op that binarizes x
    """
    #__props__ = ("rng")

    def __init__(self): 
	self.rng = np.random.RandomState(seed = 123456789)
        super(binarize, self).__init__()

    def make_node(self, x):
        # check that the theano version has support for __props__.
        #assert hasattr(self, '_props'), "Your version of theano is too old to support __props__."
        x = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]	
	temp = np.asarray(self.rng.uniform(0.,1.,x.shape), dtype=theano.config.floatX)
	z[0] = np.asarray(x > temp , dtype=theano.config.floatX)

    def grad(self, inputs, output_grads):
        return [output_grads[0]]

class GenDropoutLayer(lasagne.layers.Layer):
    
	def __init__(self, incoming, W = lasagne.init.Constant(0.5), ConvArchLearn = False,
		 **kwargs):
		super(GenDropoutLayer, self).__init__(incoming, **kwargs)
		#self._srng = RandomStreams(123456789)
		self.ConvArchLearn = ConvArchLearn

		if(len(self.input_shape) == 2 or ConvArchLearn is True):
			self.shape = [self.input_shape[1]]
		else:
			self.shape = self.input_shape[1:]

		self.W = self.add_param(W, self.shape, name="W",
				            regularizable=True, gate = True)				           
				                    	
	def get_output_for(self, input, deterministic = False, **kwargs):
		w1 = binarize()(self.W)		
		
		if(len(self.input_shape) > 2 and self.ConvArchLearn is True):
			
			if deterministic is False:
				w2 = T.reshape(w1, newshape = (1,self.shape[0],1,1))
				w3 = T.tile(w2, reps=(1,1,self.input_shape[2],self.input_shape[3]))
				output = w3 * input
			else: 
				w2 = T.reshape(self.W, newshape = (1,self.shape[0],1,1))
				w3 = T.tile(w2, reps=(1,1,self.input_shape[2],self.input_shape[3]))
				output = w3 * input
 		else:
			if deterministic is False:
				output = w1 * input
			else:
				output = self.W * input

		return output
			     



def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.
    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
             lasagne.layers.dropout(network,p=0.2), num_filters=20, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
             lasagne.layers.dropout(network,p=0.2), num_filters=5, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    network = lasagne.layers.Conv2DLayer(
             lasagne.layers.dropout(network,p=0.2), num_filters=20, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal())
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    network = lasagne.layers.Conv2DLayer(
             lasagne.layers.dropout(network,p=0.2), num_filters=10, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.HeNormal())
    network = lasagne.layers.GlobalPoolLayer(network)
    network=lasagne.layers.NonlinearityLayer(network,nonlinearity=lasagne.nonlinearities.softmax)
    #network=lasagne.layers.NonlinearityLayer(network,nonlinearity=lasagne.nonlinearities.scaled_sigmoid)
       
# A fully-connected layer of 256 units with 50% dropout on its inputs:
    #network = lasagne.layers.DenseLayer(
     #        lasagne.layers.dropout(network,p=0.2),
      #      num_units=30,
       #     nonlinearity=lasagne.nonlinearities.rectify)
    	# And, finally, the 10-unit output layer with 50% dropout on its inputs:
    #network = lasagne.layers.DenseLayer(
     #       lasagne.layers.dropout(network,p=0.2),
      #      num_units=10,
       #     nonlinearity=lasagne.nonlinearities.softmax)

    return network

from sklearn import preprocessing
from load_mnist import load_dataset
import sys
import os
import time
from matplotlib import pyplot as plt
import numpy as np
import theano
import theano.tensor as T
from sklearn.manifold import TSNE
import lasagne
from scipy.cluster.vq import kmeans
from numpy import linalg as LA
from sklearn.decomposition import PCA
import Image
from cnn import build_cnn
import scipy.io as sio
from sklearn import metrics
from theano import ProfileMode
#profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())


def ff_labels_softmax():
	X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()#loads the mnist dataset
        #X_train=np.concatenate((X_train,X_val,X_test),axis=0)[:65000,:]
        #y_train=np.concatenate((y_train,y_val,y_test),axis=0)[:65000]
	input_var = T.tensor4('inputs')
	target_var = T.ivector('targets')
	print("Building model and compiling functions...")
	network=build_cnn(input_var)
	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	ff_fn=theano.function([input_var],test_prediction)#feed forward function
	s=np.empty([10,50000])#stores the indices of samples with decreasing probabilities(p1,p2,p3...,p10)
	avg=np.empty([10])#stores the average of store
	no_of_clusters=10


	with np.load('model_dropout_test.npz') as f:
	     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(network, param_values)

	train_input=X_train

	test_prediction=ff_fn(train_input)
	
	ff_output=np.empty([50000])
	
	for i in range(50000):
        	ff_output[i]=np.argmax(test_prediction[i,:])
	'''
	for i in range(10):
		for n in range(no_of_clusters):
        		s[n,:]=np.argsort(test_prediction[:,n])[::-1]
                	avg[n]=np.mean(test_prediction[np.asarray(s[n,0:5000],int),n])
		arg=np.argmax(avg)
        	print(arg)
        	for j in range(5000):
        		ff_output[s[arg,j]]=arg
                test_prediction[:,arg]=0#equivalent to deleting the prob class        
	'''
	return ff_output
	





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
from test_models_softmax import ff_labels_softmax
from theano import ProfileMode
import timeit
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.preprocessing import OneHotEncoder
from theano.compile.nanguardmode import NanGuardMode
#profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
#profmode1 = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker()) 
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()#loads the mnist dataset
#X_train=np.concatenate((X_train,X_val,X_test),axis=0)[:65000,:]
#y_train=np.concatenate((y_train,y_val,y_test),axis=0)[:65000]
input_var = T.tensor4('inputs')
target_var = T.fmatrix('targets')
print("Building model and compiling functions...")

network=build_cnn(input_var)
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.binary_crossentropy(prediction, target_var)

lambd=10
pred=prediction.sum(axis=0)
pred=pred/pred.sum()
pred1=pred*T.log(pred)
loss = loss.mean()+lambd*pred1.mean()
#............................Dropout++ code
lambda_1= 1e-6
lambda_3= 0.

def penalty(x):
    return lambda_1 * (T.sum(T.log(T.mul(x,1. - x))) + lambda_3 * T.sum(T.log(x)))

gate_params = lasagne.layers.get_all_params(network, trainable=True, gate = True)
regularize = lasagne.regularization.apply_penalty(gate_params, penalty)
#loss = loss - regularize


lr = theano.shared(np.array(0.01, dtype=theano.config.floatX))

params = lasagne.layers.get_all_params(network, trainable=True, gate=False)
updates={}
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=lr, momentum=0.9)
updates.update(lasagne.updates.nesterov_momentum(loss, gate_params, 0.001, momentum=0.9))
#for i in range(0,8,2):
#	params[i]= lasagne.updates.norm_constraint(params[i],70)#for gradient clipping

test_prediction = lasagne.layers.get_output(network, deterministic=True)#change this to false if ff is in our algo and loss includes cross entropy

train_fn = theano.function([input_var, target_var], loss, updates=updates)
#train_fn=theano.function([input_var],loss,updates=updates)
get_pred=theano.function([input_var],pred)
get_pred1=theano.function([input_var],pred1)

a=lasagne.layers.get_all_layers(network)
b=lasagne.layers.get_output(a[7])
c=lasagne.layers.get_output(a[10])
ff_fn=theano.function([input_var],test_prediction)#feed forward function
tsne_output=theano.function([input_var],b)
gap_output=theano.function([input_var],c)

#..........................Variables used
no_of_samples=50000
no_of_clusters=10
no_of_dim=784
batchsize=500
iterate=0
mega_iter=5000
num_epochs=1




#                     1.Input
print("Shape of input:{}".format(np.shape(X_train)))
train_input=X_train
target=y_train

inputs=np.resize(X_train,(50000,784))#reshaping for projecting on 2D
m=np.mean(inputs,axis=0)
print(np.shape(m))
#means=np.empty([50000,784])
#for i in range(50000):
#	means[i,:]=m[:]
min_max_scaler = preprocessing.MinMaxScaler()
inputs=min_max_scaler.fit_transform(inputs)
input_pca=inputs
#inputs=preprocessing.scale(inputs,axis=1)
train_input=np.resize(inputs,[50000,1,28,28])#mean subtracted and used for training

#                         2.K-Means Output
#......................Extracting the kmeans labels from a mat file

a=sio.loadmat('kmeans_labels')
kmeans_target=a['id']
kmeans_target=kmeans_target-1
kmeans_target=kmeans_target[0:no_of_samples]

#         3.Neural network output

train_input=train_input.astype(np.float32)
print('CNN')
ff_output=np.empty([no_of_samples])
count=np.empty([no_of_clusters])#count of occurence of each cluster members
test_prediction = ff_fn(train_input)
print("test prediction shape:{}".format(np.shape(test_prediction)))

s=np.empty([no_of_clusters,no_of_samples])#stores the indices of samples with decreasing probabilities(p1,p2,p3...,p10)
avg=np.empty([no_of_clusters])#stores the average of store
score_all=np.empty([mega_iter])#stores the purity values for y-axis
labels_all=np.empty([no_of_samples,mega_iter])
iterate_all=np.arange(mega_iter)#stores the MI values for x-axis
'''
for i in range(no_of_samples):
	ff_output[i]=np.argmax(test_prediction[i,:])
'''

for i in range(no_of_clusters):
	for n in range(no_of_clusters):
		s[n,:]=np.argsort(test_prediction[:,n])[::-1]#stores the argument of elements of test_prediction in descending order
		
		avg[n]=np.mean(test_prediction[np.asarray(s[n,0:no_of_samples/no_of_clusters],int),n])	
		
	arg=np.argmax(avg)
	print(arg)
	for j in range(no_of_samples/no_of_clusters):
		ff_output[s[arg,j]]=arg
		
	test_prediction[:,arg]=-1#equivalent to deleting the prob class        
	

t=ff_output

for i in range(no_of_clusters):
	count[i]=np.count_nonzero(t==i)
for i in range(no_of_clusters):
	print("Cluster {} Count :{}".format(i+1,count[i]))

enc = OneHotEncoder()
t=t.reshape(-1,1)
t=enc.fit_transform(t).toarray()

for i in range(no_of_samples):
	for j in range(no_of_clusters):
		if(t[i,j]==1):
			t[i,j]=0.4
		else:
			t[i,j]=0.6/9

t=t.astype(np.float32)

#t=kmeans_target
#t=np.reshape(t,[50000])
#print(np.shape(t))
#t=t.astype(np.int32)
lambd1=0
t2=np.empty([no_of_samples,no_of_clusters])
T=0.1
incorrect=np.empty([no_of_samples])
while (iterate<mega_iter):
		
		print('Mega iteration:{}'.format(iterate+1)) 
		for epoch in range(num_epochs):
                	train_err = 0
        		train_batches = 0
        		print('Epoch:{}'.format(epoch+1))
			start=timeit.default_timer()
        		for batch in iterate_minibatches(X_train, t, batchsize, shuffle=True):
            			inputs, targets = batch
           			train_err += train_fn(inputs, targets)
				#train_err+=train_fn(inputs)
           		 	train_batches += 1
			stop=timeit.default_timer()
			print('Time Difference:{}'.format(stop-start))
        		print('Training Loss:{}'.format(train_err/train_batches))
		#profmode.print_summary()
		print('Saving models...')
		np.savez('model_dropout_test.npz', *lasagne.layers.get_all_param_values(network))
		ff_output=ff_labels_softmax()#target labels assigned to feed forward output
        	
		t=ff_output
		for i in range(no_of_clusters):
			count[i]=np.count_nonzero(t==i)
		for i in range(no_of_clusters):
			print("Cluster {} Count :{}".format(i+1,count[i]))
		#p=get_pred(X_train)
		#p1=get_pred1(X_train)
		#print(p)
		#print(p1)
		#print("Shape of pred:{},Shape of pred1:{}".format(np.shape(p),np.shape(p1)))
		score=metrics.normalized_mutual_info_score(y_train,t)
		ff_output=ff_output.astype(np.int32)
		labels_all[:,iterate]=ff_output[:]#50000  x 200
		t1=np.eye(no_of_clusters)[ff_output]
		#t=t.reshape(-1,1)
                #t=enc.fit_transform(t).toarray()
	
		
		print("		Metric score:{}".format(score))
		score_all[iterate]=score
		if(iterate!=0):
                        print("		Improvement:{}".format(score_all[iterate]-score_all[iterate-1]))
			acc=accuracy_score(labels_all[:,iterate],labels_all[:,iterate-1])
			print("Accuracy:{}".format(acc))
                        #f1=f1_score(labels_all[:,iterate],labels_all[:,iterate-1])
                        #print("F1_score :{}".format(f1))
                        index=(labels_all[:,iterate]==labels_all[:,iterate-1])
                        for i in range(np.size(index)):
                            if(index[i]==True):
                                incorrect[i]=0
                            else:
                                incorrect[i]=1

			for i in range(no_of_samples):
                            if(incorrect[i]==0):
                                for j in range(no_of_clusters):
					if(t1[i,j]==1):
						t1[i,j]=acc
					else:
						t1[i,j]=(1-acc)/9
                            else:
                                for j in range(no_of_clusters):
                                    if(t1[i,j]==1):
                                        t1[i,j]=acc/2
                                    else:
                                        t1[i,j]=(1-(acc/2))/9
		else:
			for i in range(no_of_samples):
				for j in range(no_of_clusters):
					if(t1[i,j]==1):
						t1[i,j]=0.4
					else:
						t1[i,j]=0.6/9

		f1=gap_output(X_train)
                #print('Temperature:{}'.format(T))
		temp=np.exp(np.array(f1)/T)
		temp_sum=temp.sum(-1)
		for i in range(no_of_samples):
			t2[i]=temp[i,:]/temp_sum[i]
	
		#print("Temp_sum:{}".format(temp_sum[0:10]))	
		t=(1-lambd1)*t1+lambd1*t2
		#t=t2
                
		index_low=t<0.1
		t[index_low]=0.1
		index_high=t>1-0.1
		t[index_high]=1-0.1
		#print(t1[0,:])
		#print(t2[0,:])
		#print(t[0,:])
                
		t=t.astype(np.float32)
		'''
		if(iterate% 20 ==0):
			pca=PCA(n_components=2)
			data=pca.fit_transform(input_pca)
			fig1 = plt.figure()
			ax1=fig1.add_subplot(111)
			for i in range(1000):
				if(t[i]==0):
					c0=ax1.scatter(data[i,0],data[i,1],s=20,c='r',marker='o')
				elif(t[i]==1):
					c1=ax1.scatter(data[i,0],data[i,1],s=20,c='b',marker='+')
				elif(t[i]==2):
                                        c2=ax1.scatter(data[i,0],data[i,1],s=20,c='g',marker='*')
                        
				elif(t[i]==3):
                                        c3=ax1.scatter(data[i,0],data[i,1],s=20,c='c',marker='o')
                        
				elif(t[i]==4):
                                        c4=ax1.scatter(data[i,0],data[i,1],s=20,c='m',marker='o')
                        
				elif(t[i]==5):
                                        c5=ax1.scatter(data[i,0],data[i,1],s=20,c='y',marker='o')
                        
				elif(t[i]==6):
                                        c6=ax1.scatter(data[i,0],data[i,1],s=20,c='k',marker='o')
                        
				elif(t[i]==7):
                                        c7=ax1.scatter(data[i,0],data[i,1],s=20,c='b',marker='o')
                        
				elif(t[i]==8):
                                        c8=ax1.scatter(data[i,0],data[i,1],s=20,c='r',marker='+')
                        
				elif(t[i]==9):
                                        c9=ax1.scatter(data[i,0],data[i,1],s=20,c='g',marker='+')
			
			ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),scatterpoints=1,
          				ncol=5, fancybox=True, shadow=True)
			ax1.legend([c0,c1,c2,c3,c4,c5,c6,c7,c8,c9],['1','2','3','4','5','6','7','8','9','10'])
			fig1.savefig('PCA'+str(iterate)+'.png')
			
			
                        model = TSNE(n_components=2, random_state=0)#TSne plot
                        np.set_printoptions(suppress=True)
                        input_tsne=tsne_output(train_input)
                        input_tsne=np.resize(input_tsne,[50000,720])
                        data=model.fit_transform(input_tsne[0:1000])
                        fig2=plt.figure()
                        ax2=fig2.add_subplot(111)

                        for i in range(1000):
                                if(t[i]==0):
                                        c0=ax2.scatter(data[i,0],data[i,1],s=20,c='r',marker='o')
                                elif(t[i]==1):
                                        c1=ax2.scatter(data[i,0],data[i,1],s=20,c='b',marker='+')
                                elif(t[i]==2):
                                        c2=ax2.scatter(data[i,0],data[i,1],s=20,c='g',marker='*')

                                elif(t[i]==3):
                                        c3=ax2.scatter(data[i,0],data[i,1],s=20,c='c',marker='o')

                                elif(t[i]==4):
                                        c4=ax2.scatter(data[i,0],data[i,1],s=20,c='m',marker='o')

                                elif(t[i]==5):
                                        c5=ax2.scatter(data[i,0],data[i,1],s=20,c='y',marker='o')

                                elif(t[i]==6):
                                        c6=ax2.scatter(data[i,0],data[i,1],s=20,c='k',marker='o')
	                        elif(t[i]==7):
                                        c7=ax2.scatter(data[i,0],data[i,1],s=20,c='b',marker='o')

                                elif(t[i]==8):
                                        c8=ax2.scatter(data[i,0],data[i,1],s=20,c='r',marker='+')

                                elif(t[i]==9):
                                        c9=ax2.scatter(data[i,0],data[i,1],s=20,c='g',marker='+')

                        iax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),scatterpoints=1,
                                        ncol=5, fancybox=True, shadow=True)
                        ax2.legend([c0,c1,c2,c3,c4,c5,c6,c7,c8,c9],['1','2','3','4','5','6','7','8','9','10'])
                      	fig2.savefig('TSNE'+str(iterate)+'.png')
		'''	
		iterate=iterate+1

print(ff_output)
#feature=tsne_output(train_input)
#feature=np.resize(feature,[50000,720])
#sio.savemat('feature.mat',{'feature':feature})

np.savez('model_loss_penalty.npz', *lasagne.layers.get_all_param_values(network))

sio.savemat('score_all.mat',{'score_all':score_all})

sio.savemat('ff_output.mat',{'ff_output':t})

print('K-Means Performance:')

kmeans_target=np.reshape(kmeans_target,[no_of_samples])
score=metrics.normalized_mutual_info_score(y_train,kmeans_target)
homo=metrics.homogeneity_score(y_train, kmeans_target)
comp=metrics.completeness_score(y_train, kmeans_target)
v_meas=metrics.v_measure_score(y_train, kmeans_target)
print("Metric score:{} Homogeniety:{} Completeness:{} V_Measure:{}".format(score,homo,comp,v_meas))

#print('CNN Performance')
'''
score=metrics.adjusted_mutual_info_score(y_train,t)
homo=metrics.homogeneity_score(y_train, ff_output)
comp=metrics.completeness_score(y_train, ff_output)
v_meas=metrics.v_measure_score(y_train,ff_output)
print("Metric score:{} Homogeniety:{} Completeness:{} V_Measure:{}".format(score,homo,comp,v_meas))

fig3=plt.figure()
plt.plot(iterate_all,score_all,marker='o',color='r',linestyle='-')
plt.ylabel('NMI')
plt.xlabel('Meta Iterations')
fig3.savefig("AMI.png")
'''

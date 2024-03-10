import numpy as np

"""
A pseudo code of Stochastic Gradient Descent (SGD) algorithm.
"""
def sgd(data,label,noe=100,mini_batch_size=32,sampling,weigh_init,loss_func,gradient,lr=0,001):
	w=weight_init()
	l=[]
	number_of_samples = data.shape[0]
	for t in range(noe):
		for k in range(number_of_samples//mini_batch_size)
			samples=sampling()
			batch_data=data[samples]
			batch_label=label[samples]
			predict_label = predict(batch_data,W,b)
			dw=gradient(loss_func,batch_data,batch_label,W)
			W-=learning_rate * dw
	return W


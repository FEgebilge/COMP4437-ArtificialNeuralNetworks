# Regularization & Optimization

> Problem!
> Loss functions encourage good performance on *training* data but we really care about *test* data
> 

**Overfitting:** A model is overfit when it performs too well on the training data, and has poor performance for unseen data.

##### Beyond Training Error
$$
L(W)=\frac{1}{N}\sum_{i=1}^N Li(f(x_i,W)y_i)+\lambda R(W)
$$

$\lambda$ is a hyperparameter giving re regularization strength

**Simple Examples**
L2 regularization: $R(W)\sum_{k,l}W^2_{k,l}$

L1 regularization: $R(W)\sum_{k,l}|W_{k,l}|$

>[!WARNING]
> Finding a good W: Data Loss + Regularization
> Data Loss: Model predictions should match training data
> Regularization: Prevent the model from doing too well on training data
> Second one has certainty at some points unnaturally. At these points it should return uncertain results. This causing it work bad even it's loss is much lower.
><img width="655" alt="unnatural_cliff_regularization" src="https://github.com/FEgebilge/COMP4437-ArtificialNeuralNetworks/assets/93092469/5ac235f0-0dbd-4c75-a502-948ac1b2d9fc">
>
> 	This problem called as **memorization**

###### Expressing Preferences
$x=[1,1,1,1]$

$W_1=[1,0,0,0]$

$W_2=[0.25,0.25,0.25,0.25]$ -> *Preferred*

$W_1^TX=W_2^TX=1$  -> Same predictions, so data loss will always be the same

$$
R(W)=\sum_{k,l}W_{k,l}^2
$$

>L2 regularization prefers weight to be "spread out"
___
#### Optimization
$$
W^*=argminL(W)
$$

>"Training a neural net is basically to find a minimum(hopefully the global) of some complex function."

**Problem:** Cannot evaluate the loss function at every possible coordinate, it is expensive so, basically we're blind.
*Ideas:*
1. Random Search(Bad Idea)
2. Follow the Slope

##### Follow the slope
- In 1 dimension, the **derivative** of a function gives the slope
- In multiple dimensions, the **gradient** is the vector of (partial derivatives) along each dimension
	The slope in any direction is the **dot product** of the direction with gradient.
	The direction of steepest descent is the **negative gradient**
<img width="634" alt="calculate_gradient" src="https://github.com/FEgebilge/COMP4437-ArtificialNeuralNetworks/assets/93092469/feb3e736-aaa1-4864-8fb2-b67fecc0fb11">

##### Computing Gradients
- **Numeric gradient:** approximate, slow, easy to write
- **Analytic gradient:** exact, fast, error-prone
  
	*In practice:* Always use analytic gradient, but check implementation with numerical gradient. This is called *gradient check*. `torch.autograd.gradcheck`
  
##### Gradient Descent
Iteratively step in the direction of the negative gradient(direction of local steepest descent)
```python
# Vanilla gradient descent
w = initialize weights()
for t in range(num_steps):
	dw = compute_gradient(loss_fn, data, w)
	w -= learning_rate * dw
```

**Hyperparameters:**
- Weight initialization method
- Number of steps
- Learning rate -> how much you want to step (step size)

##### Batch Gradient Descent
$$
\nabla_W L(W)=\frac{1}{N}\sum_{i=1}^N\nabla_WL_i(x_i,y_i,W)+\lambda R(W)
$$

$x_i,y_i:$ training pairs

$L_i:$ Loss function

- Full sum is expensive when N is large!
  
	This leads us to use a minibatch.

##### Stochastic Gradient Descent(SGD)
>Approximate the loss function and gradient by drawing small subsamples from full training dataset.

**Two new hyperparameters:**
- Batch size
- Data sampling
```python
w = initialize_weights()
for t in range(num_steps):
	minibatch = sample_data(data, batch_size)
	dw = compute_graditent(loss_fn,minibatch,w)
	w -= learning_rate * dw
```

##### Problems with SGD
- If loss changes quickly in one direction and slowly in the other one
- If loss function has a **local minimum** or **saddle point**
- If gradients are noisy (they came from mini batches)

##### To overcome those problems
###### SGD + Momentum
**SGD:** 
$x_{t+1}=x_t-\alpha \nabla f(x_t)$

**SGD + Momentum:** 
There is a "velocity" vector running as a mean of gradients (rho gives "friction")
$V_{t+1}=\rho V_t + \nabla f(x_t)$
$x_{t+1}=x_t-\alpha V_{t+1}$

<img width="326" alt="SGD+Momentum Update" src="https://github.com/FEgebilge/COMP4437-ArtificialNeuralNetworks/assets/93092469/d4d43d11-1434-495b-9f9b-9f95fada086e">

Combine gradient at current point with velocity to get step used to update weights

---
#### Links
- [Self study](https://cs231n.github.io/optimization-1/)
- [Interactive web demo](http://vision.stanford.edu/teaching/cs231n-demos/linear-classify/)

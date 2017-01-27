import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  num_classes = W.shape[1]
  num_train = X.shape[0]

  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
  	label = y[i]
  	score_vector = np.dot(X[i],W)
  	exp_score_vector = np.exp(score_vector)
  	denominator = np.sum(exp_score_vector)
  	normalised_vector = exp_score_vector/denominator
  	loss += -np.log(normalised_vector[label])
  pass

  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)

  for i in xrange(num_train):
  	label = y[i]
  	score_vector = np.dot(X[i],W)
  	exp_score_vector = np.exp(score_vector)
  	denominator = np.sum(exp_score_vector)
  	normalised_vector = exp_score_vector/denominator

  	for j in xrange(num_classes):
  		if j == label :
  			dW[:,j] += -(1-normalised_vector[label])* X[i]
  		else :
  		   dW[:,j] += normalised_vector[j] * X[i]

  dW /= num_train
  dW += reg * W		   	   

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.

  num_classes = W.shape[1]
  num_train = X.shape[0]


  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score_mat = np.dot(X,W)
  exp_score_mat = np.exp(score_mat)
  denominator_vector = np.sum(exp_score_mat,axis = 1)
  normalised_mat = exp_score_mat / np.tile(np.reshape(denominator_vector,(len(denominator_vector),1)),(1,num_classes))
  vector_loss  =  normalised_mat[ np.arange(num_train) , y  ]
  loss = np.sum(-np.log(vector_loss))

  prob_mat = normalised_mat
  prob_mat[np.arange(num_train),y] -=  1

  dW = np.dot(X.T,prob_mat)/num_train
  dW += reg*W

  loss /= num_train
  loss += 0.5*reg*np.sum(W*W)

  


  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


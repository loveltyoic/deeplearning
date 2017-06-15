import numpy as np
from random import shuffle
# from past.builtins import xrange
import math
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
  loss = 0.0
  dW = np.zeros_like(W)
    
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_train):
        f = X[i].dot(W)
        f = f - np.max(f)
        loss -= math.log(math.exp(f[y[i]]) / np.sum(np.exp(f)))
        sum_i = np.sum(np.exp(f))
        for j in range(num_class):
            if j == y[i]:
                dW[:, j] += (np.exp(f[j]) / sum_i - 1) * X[i]
            else:
                dW[:, j] += (np.exp(f[j]) / sum_i) * X[i]
                
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  dW = dW / num_train + reg * W
  loss = loss / num_train + 0.5 * reg * np.sum(W * W)
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  f = X.dot(W)
  f = f - np.max(f)
  mat = np.exp(f)
  loss = -np.sum(np.log(mat[np.arange(num_train), y] / np.sum(mat, axis=1))) / num_train + 0.5 * reg * np.sum(W * W)
  dmat = np.zeros((num_train, num_class))
  dmat[np.arange(num_train), y] = 1
  dmat = mat / np.expand_dims(np.sum(mat, axis=1), 1) - dmat
  dW = X.T.dot(dmat) / num_train + reg * W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    dW = np.zeros_like(W) #(3073,10)
    num_dev= X.shape[0]#500
    num_class=W.shape[1]#10
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(num_dev):
      score=X[i].dot(W)# (10,) 
      score-=np.max(score) #logC=-max_j(f_j) in the lecture module for numerical stability

      softmax=np.exp(score[y[i]])/np.sum(np.exp(score)) #() 
      loss-=np.log(softmax)

      p= lambda k: np.exp(score[k])/np.sum(np.exp(score)) #intermediate value p for gradient calculation. p means softmax function for a given class j.

      for k in range(num_class):
        p_k=p(k) # softmax output for class k.
        dW[:,k]+=(p_k-(k==y[i]))*X[i]
        
    
    loss/=num_dev
    loss+=reg*np.sum(W*W)
    dW/=num_dev
    dW+=2*reg*W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_dev= X.shape[0]#500
    num_class=W.shape[1]#10
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    score=X.dot(W) #(500,10)
    score-=np.max(score,axis=1).reshape(-1,1) #(500,10)
    sum_score=np.sum(np.exp(score),axis=1).reshape(-1,1) #(500,1)
    p=np.exp(score)/sum_score #(500,10)
    loss = np.sum(-np.log(p[np.arange(num_dev), y]))

    mask=np.zeros_like(p) #(500,10)
    mask[np.arange(num_dev),y]=1 #1 for correct classes
    intermediate=p-mask #(500,10)
    dW=X.T.dot(intermediate)

    loss/=num_dev
    loss+=reg*np.sum(W*W)
    dW/=num_dev
    dW+=2*reg*W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

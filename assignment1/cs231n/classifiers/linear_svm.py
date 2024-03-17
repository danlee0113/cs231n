from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.# Here, (3073,10)
    - X: A numpy array of shape (N, D) containing a minibatch of data. # Here, (500,3072)
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1] # C
    num_train = X.shape[0] # N
    loss = 0.0
    for i in range(num_train):# this process follows the notation from the slides
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]] #eqivalent to s_yi in slides
        for j in range(num_classes):
          margin = scores[j] - correct_class_score + 1 
          if j == y[i]: # if the correct class== the class we are paying attention
            continue
          if margin>0:
            loss+=margin
            dW[:,y[i]]-=X[i]
            dW[:,j]+=X[i]


    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W) # L2 loss

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW/=num_train
    dW+=2*reg*W
   
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_train = X.shape[0] # N
    num_classes = W.shape[1] # C
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores=X.dot(W) #(500,10) -> scores of all classes for 500 samples.
    correct_y_scores=scores[np.arange(num_train), y].reshape(num_train,1)
    margin=np.maximum(scores-correct_y_scores+1,0)
    margin[np.arange(num_train), y] = 0 # Correct class=0(we don't consider correct classes when calculating the loss)
    loss=np.sum(margin)/num_train
    loss += reg * np.sum(W * W)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW=(margin>0).astype(int) #111...0111 -> correct class has label 0.
    dW[np.arange(num_train),y]-=dW.sum(axis=1)#dW has shape (3073,10)-> so sum w.r.t class dimension. we subtract (classes-1) -> 1 being correct class.
    dW=X.T.dot(dW)/num_train +2*reg*W
   
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

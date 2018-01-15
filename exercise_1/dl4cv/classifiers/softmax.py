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
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    C = W.shape[1]
    N = X.shape[0]

    for i in range(N):
        
        scores = X[i].dot(W)

        scores -= max(scores)   #For numerical stability we subtract max scores from scores

        correct_class_score = scores[y[i]]

        sum_exps = sum(np.exp(scores))

        loss += -correct_class_score + np.log(sum_exps)

        #Now for the derivative
        for j in range(C):
            
            p = np.exp(scores[j]) / sum_exps

            dW[:, j] += (p - (j == y[i])) * X[i]

    loss /= N
    dW /= N

    #Add regularisation
    loss += 0.5 * reg * np.sum(W*W)
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
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    C = W.shape[1]
    N = X.shape[0]

    scores = X.dot(W)
    scores -= np.max(scores, axis=1).reshape(-1, 1)    #Avoid numerical stability
    softmax_output = np.exp(scores)/np.sum(np.exp(scores), axis = 1).reshape(-1,1)
    loss = -np.sum(np.log(softmax_output[range(N), list(y)]))

    loss /= N
    loss += 0.5 * reg * np.sum(W*W)

    dS = softmax_output.copy()
    dS[range(N), list(y)] += -1
    dW = (X.T).dot(dS)
    dW = dW/N + reg* W


    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


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
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        # print("X的shape" + str(X.shape))
        # print("W的shape"+str(W.shape))
        # print(X.dot(W).shape)
        scores = X[i].dot(W)
        # print("scores的shape"+str(scores.shape))
        scores -= np.max(scores)
        loss_i = -scores[y[i]] + np.log(sum(np.exp(scores)))
        loss += loss_i

        for j in range(num_classes):
            softmax_out = np.exp(scores[j]) / sum(np.exp(scores))
            if j == y[i]:
                dW[:, j] += (-1 + softmax_out) * X[i]
            else:
                dW[:, j] += softmax_out * X[i]
    loss /= num_train
    loss += 0.5 * reg * np.sum(W*W)
    dW /= num_train + reg * W

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    # num_train = X.shape[0]
    # num_classes = W.shape[1]
    # scores = X.dot(W)
    # scores -= np.max(scores, axis=1).reshape(-1, 1)
    #
    # softmax_output = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape((-1, 1))
    #
    # loss = -np.sum(np.log(softmax_output[range(num_train), list(y)])) # 计算的是正确分类y_i的损失
    # loss /= num_train
    # loss += 0.5*reg*np.sum(W*W)
    #
    # dS = softmax_output.copy()
    # dS[range(num_train), list(y)] += -1
    # dW = (X.T).dot(dS)
    # dW /= num_train + reg * W
    ######################
    num_samples = X.shape[0]
    num_classes = W.shape[1]
    score = X.dot(W)  # N by C
    prob = score - np.max(score, axis=1, keepdims=True)
    prob = np.exp(prob) / np.sum(np.exp(prob), axis=1, keepdims=True)  # N by C, Mat of P
    loss = np.sum(-1 * np.log(prob[range(num_samples), y]))
    prob[range(num_samples), y] -= 1  # j == y[i] , dw = (P_ij - 1)Xi
    dW = X.T.dot(prob)  # (D by N)(N by C) = D by C

    loss /= num_samples
    dW /= num_samples

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

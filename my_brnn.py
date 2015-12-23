#! /usr/bin/env python
"""
file: my_brnn.py
author: thomas wood (thomas@wgapl.com)
description: A quick and dirty Bidirectional Recurrent Layer in Python.
"""

import numpy as np
from numpy import tanh
from numpy.random import random
from string import printable


def activation(z, method="tanh"):
    """
    Defaults to "tanh".
    Probably shouldn't ever neglect to use that, but whatever.
    """
    if method == "tanh":
        return tanh(z)
    elif method == "linear":
        return z
    elif method == "sigmoid":
        return 1./(1.+np.exp(-z))



def gen_bag_hashtable():
    N = len(printable)
    table = {}
    for k in range(N):
        table[printable[k]] = k
    return table

def make_wordvector(s, table):
    N = len(printable)
    L = len(s)
    a = np.zeros((N,L))
    for k in range(L):
        a[ table[ s[k] ], k ] = 1
    return a

def make_string(x):
    s = []
    for k in range(x.shape[1]):
        s.append(printable[np.argmax(x[:,k])])
    return ''.join(s)


class BRNNLayer:
    def __init__(self,n_in, n_hidden, n_out, params, eps):
        """
        There are six weight matrices

        W_xfh, W_xbh -- n_in X n_hidden input matrices
        W_fhh, W_bhh -- n_hidden X n_hidden recurrent matrices
        W_yfh, W_ybh -- n_hidden X n_out output matrices

        and three bias vectors

        b_f -- n_hidden X 1 forward bias vector
        b_b -- n_hidden X 1 backward bias vector
        b_y -- n_out X 1 output bias vector

        """
        # basic layer dimensions -- determines size of weight matrices
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out

        ###---!!! MODEL PARAMETERS !!!---###
        x_ind = 2 * n_in * n_hidden # num params in W_x
        self.W_x = params[:x_ind].reshape((2*n_hidden, n_in)) # Concatenated input weight matrices

        h_ind = x_ind + 2 * n_hidden*n_hidden # n_params in W_h
        self.W_h = params[x_ind:h_ind].reshape((2*n_hidden, n_hidden)) # Concatenated recurrent weight matrices

        y_ind = h_ind + 2 * n_hidden * n_out # n_params in W_y
        self.W_y = params[h_ind:y_ind].reshape((n_out,2*n_hidden))

        self.bias = params[y_ind:] # rest of parameters are biases



    def gen_sequence(self, X):
        """
        Bidirectional RNN update of output sequence based on paper @
        http://www.cs.toronto.edu/~graves/asru_2013.pdf

        So I basically keep a single function as a member of the BRNNLayer struct
        just for making a prediction based on a sequence of input data.
        Still not really OOP if you ask me, and I'm fine with that.
        """
        n_in = self.n_in
        n_hidden = self.n_hidden
        n_out = self.n_out
        T = X.shape[1] # length of input/output sequence

        # For the reader:
        # b_f = self.bias[:n_hidden] -- forward bias
        # b_b = self.bias[n_hidden:2*n_hidden] -- backward bias
        # b_y = self.bias[2*n_hidden:] -- output bias

        # Many values need to calculate hidden states only depend on quantities in X
        Wx = np.dot(self.W_x, X) # Compute the values from input connections

        # Initializing hidden state matrix
        #      _                            _
        #     |                              |
        #     |  hf_0  hf_1  hf_2 ... hf_T-1 |
        # H = |                              |
        #     |  hb_0  hb_1  hb_2 ... hb_T-1 |
        #     |_                            _|
        #
        # where hf_0..T-1 are column vectors representing forward hidden states
        # and hb_0..T-1 are column vectors representing backward hidden states
        H = np.zeros((2*n_hidden,T))

        # First and last sequences don't have a predecessor, so they get special
        # consideration. Prime the pump.
        H[:n_hidden,0] = activation(Wx[:n_hidden,0]+self.bias[:n_hidden])
        H[n_hidden:,T-1] = activation(Wx[n_hidden:,T-1]+self.bias[n_hidden:2*n_hidden])

        # Iterate over the sequence forward and backwards.
        for k in range(1,T):
            # FORWARD: calculate forard hidden values according to rules in paper.
            H[:n_hidden,k] = activation(Wx[:n_hidden,k] + \
            np.dot(self.W_h[:n_hidden,:],H[:n_hidden,k-1]) + self.bias[:n_hidden])

            # BACKWARD: populate backward hidden states in reverse order across sequence
            H[n_hidden:,T-1-k] = activation(Wx[n_hidden:,T-1-k] + \
            np.dot(self.W_h[n_hidden:,:],H[:n_hidden,T-k]) + self.bias[n_hidden:2*n_hidden])

        # Now that H has been calculated, finding the layer output is straightforward
        return activation(np.dot(self.W_y,H) + np.tile(self.bias[2*n_hidden:], (T,1)).T,
            method="linear")


def rudimentary_test():
    """
    Very simple test of BRNNLayer functionality. I'm training a DQN for
    Space Invaders right now and I don't really want to get into any training
    until my GPU is free for all the matrix multiplication.

    Right now this is just a fun example of how to multiply random numbers
    to get more random numbers. I might add in some objective costs along with
    some optimization routines, but I would likely make a new repository for
    my optimization function.
    """

    s = """0 a is the quick fox who jumped over the lazy brown dog's new sentence."""
    table = gen_bag_hashtable()

    v = make_wordvector(s, table)

    n_in, T = v.shape
    n_out = n_in
    n_hidden = 100 # Learn a more complex representation?
    eps = 0.1


    n_params = 2*n_in*n_hidden + \
    2*n_hidden*n_hidden + \
    2*n_out*n_hidden + \
    2*n_hidden+n_out

    params1 = eps*(2*random(n_params,)-1.)
    params2 = eps*(2*random(n_params,)-1.)
    params3 = eps*(2*random(n_params,)-1.)

    brnn1 = BRNNLayer(n_in,n_hidden,n_in,params1, eps)
    brnn2 = BRNNLayer(n_in,n_hidden,n_in,params2, eps)
    brnn3 = BRNNLayer(n_in,n_hidden,n_in,params3, eps)

    y1 = brnn1.gen_sequence(v)
    y2 = brnn2.gen_sequence(y1)
    y3 = brnn3.gen_sequence(y2)

    s1 = make_string(y3)
    print s1

if __name__ == "__main__":
    rudimentary_test()

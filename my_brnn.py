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


def sigmoid(z):
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
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out

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
        """
        n_in = self.n_in
        n_hidden = self.n_hidden
        n_out = self.n_out
        T = X.shape[1]

        b_f = self.bias[:n_hidden]
        b_b = self.bias[n_hidden:2*n_hidden]
        b_y = self.bias[2*n_hidden:]
        if X.shape[0] != n_in:
            return "Size of feature space of data and network disagree"

        # Only depends on values in X
        Wx = np.dot(self.W_x, X) # Compute the values of input connections

        # Initializing hidden state matrix
        H = np.zeros((2*n_hidden,T))
        H[:n_hidden,0] = tanh(Wx[:n_hidden,0]+b_f)
        H[n_hidden:,T-1] = tanh(Wx[n_hidden:,T-1]+b_b)
        for k in range(1,T):
            H[:n_hidden,k] = tanh(Wx[:n_hidden,k] + \
            np.dot(self.W_h[:n_hidden,:],H[:n_hidden,k-1]) + b_f)
            print T-k+1
            H[n_hidden:,T-k-1] = tanh(Wx[n_hidden:,T-k-1] + \
            np.dot(self.W_h[n_hidden:,:],H[:n_hidden,T-k]) + b_b)

        B_y = np.tile(b_y, (T,1)).T # create a n_out X T matrix

        # G for Guess
        print H.shape
        print self.W_y.shape
        return np.dot(self.W_y,H) + B_y


def rudimentary_test():
    s = """0 a is the quick fox who jumped over the lazy brown dog's new sentence."""
    table = gen_bag_hashtable()
    v = make_wordvector(s, table)
    s0 = make_string(v)
    # print s0
    # print v
    n_in, T = v.shape
    n_out = n_in
    n_hidden = 100 # Learn a more complex representation?
    eps = 0.01
    print n_in, T

    n_params = 2*n_in*n_hidden + \
    2*n_hidden*n_hidden + \
    2*n_out*n_hidden + \
    2*n_hidden+n_out

    params1 = random(n_params,)

    brnn1 = BRNNLayer(n_in,n_hidden,n_in,params1, eps)

    g = brnn1.gen_sequence(v)
    s1 = make_string(g)
    print s1

if __name__ == "__main__":
    rudimentary_test()

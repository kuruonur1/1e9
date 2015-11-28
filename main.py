import lasagne, theano, numpy as np, logging
from theano import tensor as T
import argparse

import model

# random.seed(0)
rng = np.random.RandomState(1234567)
lasagne.random.set_rng(rng)

# param_names = ['activation','n_hidden','drates','opt','lr','norm']
def get_arg_parser():
    parser = argparse.ArgumentParser(prog="main")
    
    parser.add_argument("--activation", default='relu', choices=['sigmoid','tanh','relu'])
    parser.add_argument("--n_hidden", default=[128], nargs='+', type=int, help="number of neurons in each hidden layer")
    parser.add_argument("--drates", default=[0, 0], nargs='+', type=float, help="dropout rates")
    parser.add_argument("--opt", default="adam", help="optimization method: sgd, rmsprop, adagrad, adam")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--norm", default=5, type=float, help="Threshold for clipping norm of gradient")

    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = vars(parser.parse_args())

    dat = np.load('toy.npz')
    trn, dev, tst = dat['trn'], dat['dev'], dat['tst']

    dnn = model.DNN(20,10,args)

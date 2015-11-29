import lasagne, theano, numpy as np, logging
from theano import tensor as T
import argparse
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt.pyll.stochastic

import model
import prep

# random.seed(0)
rng = np.random.RandomState(1234567)
lasagne.random.set_rng(rng)

# param_names = ['activation','n_hidden','drates','opt','lr','norm']
def get_arg_parser():
    parser = argparse.ArgumentParser(prog="main")
    
    parser.add_argument("--toy", default=1, type=int, help="toy or real")
    parser.add_argument("--fepoch", default=10, type=int, help="num of epochs")

    return parser


"""
def main():
    parser = get_arg_parser()
    args = vars(parser.parse_args())
    print args

    NF, NOUT = 400, 200
    if args['toy']:
        dat = np.load('toy.npz')
        trn, dev, tst = dat['trn'], dat['dev'], dat['tst']
    else:
        trn, dev, tst = map(prep.get_dset, ('trn','dev','tst'))

    trnX, trnY = trn[:,NOUT:], trn[:,:NOUT]
    devX, devY = dev[:,NOUT:], dev[:,:NOUT]

    dnn = model.DNN(NF,NOUT,args)
    for e in range(args['fepoch']):
        tcost = dnn.train(trnX, trnY)
        dcost = dnn.predict(devX, devY)
        print dcost
"""

def create_space(max_layer_count, h_opts=[16,32]):
    MAXL=max_layer_count
    dpart = [
        dict(
            [('h%dm%d'%(l,maxl), hp.choice('h%dm%d'%(l,maxl), h_opts)) for l in range(1,maxl+1)] +
            [('d%dm%d'%(l,maxl), hp.uniform('d%dm%d'%(l,maxl), 0,1)) for l in range(0,maxl+1)]
        ) for maxl in range(1,MAXL+1)]

    space = {
            'activation' : hp.choice('activation',['sigmoid','tanh','relu']),
            'lr':hp.uniform('lr',0.001,2),
            'norm':hp.uniform('norm',0.1,100),
            'n_batch':hp.choice('n_batch',[32,64,128,256,512]),
            'dpart' : hp.choice('dpart', dpart),
    }
    return space


def main_opt():
    parser = get_arg_parser()
    args = vars(parser.parse_args())
    print args

    NF, NOUT = 400, 200
    dat = np.load('toy.npz')
    trn, dev, tst = dat['trn'], dat['dev'], dat['tst']
    trnX, trnY = trn[:,NOUT:], trn[:,:NOUT]
    devX, devY = dev[:,NOUT:], dev[:,:NOUT]

    def objective(conf):
        conf['n_hidden'] = map(lambda x:x[1], sorted((k,v) for k, v in conf['dpart'].iteritems() if k.startswith('h')))
        conf['drates'] = map(lambda x:x[1], sorted((k,v) for k, v in conf['dpart'].iteritems() if k.startswith('d')))
        conf['opt'] = 'adam'
        print 'conf:', conf

        dnn = model.DNN(NF,NOUT,conf)

        for e in range(args['fepoch']):
            tcost = dnn.train(trnX, trnY)
            dcost = dnn.predict(devX, devY)

        return {
                'loss': dcost,
                'status': STATUS_OK,}

    space = create_space(3)

    best = fmin(objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            )
    print best

def hp_test():
    import hyperopt.pyll.stochastic
    space = create_space(3)
    for e in range(5):
        space_sample = hyperopt.pyll.stochastic.sample(space)
        print space_sample 

if __name__ == '__main__':
    main_opt()

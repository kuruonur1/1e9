import lasagne, theano, numpy as np, logging
from theano import tensor as T
import argparse
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt.pyll.stochastic
from tabulate import tabulate

import model
import prep

# random.seed(0)
rng = np.random.RandomState(1234567)
lasagne.random.set_rng(rng)

# param_names = ['activation','n_hidden','drates','opt','lr','norm']
def get_arg_parser():
    parser = argparse.ArgumentParser(prog="main")
    
    parser.add_argument("--toy", default=1, type=int, help="toy or real")
    parser.add_argument("--activation", default='sigmoid', choices=['sigmoid','tanh','relu','elu'])
    parser.add_argument("--n_hidden", default=[512], type=int, nargs='+')
    parser.add_argument("--drates", default=[0,0], type=float, nargs='+')
    parser.add_argument("--n_batch", default=128, type=int)
    parser.add_argument("--bnorm", default=0, type=int)
    parser.add_argument("--opt", default='sgd')
    parser.add_argument("--lr", default=.1, type=float)
    parser.add_argument("--norm", default=100, type=float)
    parser.add_argument("--fepoch", default=100, type=int, help="num of epochs")
    parser.add_argument("--log", default='nothing', help="log file name")

    return parser

def setup_logger(args):
    import socket
    host = socket.gethostname().split('.')[0]
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    shandler = logging.StreamHandler()
    shandler.setLevel(logging.INFO)
    logger.addHandler(shandler);

    if len(args['log']) > 0 and args['log'] != 'nothing':
        ihandler = logging.FileHandler('logs/{}.log'.format(args['log']), mode='w')
        ihandler.setLevel(logging.INFO)
        logger.addHandler(ihandler);


def main():
    parser = get_arg_parser()
    args = vars(parser.parse_args())
    setup_logger(args)

    logging.info(tabulate([args],headers='keys',tablefmt='plain'))

    NF, NOUT = 400, 200
    logging.info('loading data...')
    if args['toy']:
        dat = np.load('data/toy.npz')
        trn, dev, tst = dat['trn'], dat['dev'], dat['tst']
    else:
        trn, dev, tst = map(prep.get_dset, ('trn','dev','tst'))

    logging.info('loading data done.')

    trnX, trnY = trn[:,NOUT:], trn[:,:NOUT]
    devX, devY = dev[:,NOUT:], dev[:,:NOUT]


    dnn = model.DNN(NF,NOUT,args)
    costs = []
    for e in range(args['fepoch']):
        tcost = dnn.train(trnX, trnY)
        dcost, pred = dnn.predict(devX, devY)
        costs.append(dcost)
        print 'dcost: {} pred: {} pred avg norm: {} truth avg norm: {}'.format(dcost, pred.shape, np.mean(np.linalg.norm(pred,axis=1)), np.mean(np.linalg.norm(devY,axis=1)))
        """
        t, p = devY[5,:], pred[5,:]
        print t
        print p
        print np.sum((t-p)**2)/2
        break
        """

    logging.info('dcost with best model: {}'.format(min(costs)))


def best2mparams(best, opts):
    mparams = {}
    mparams.update((e, opts[e][best[e]]) for e in ['activation', 'n_batch', 'opt'])

    mparams.update((e, best[e]) for e in ['norm','lr'])

    n_hidden_ind = map(lambda x:x[1], sorted((k,v) for k, v in best.iteritems() if k.startswith('h')))
    mparams['n_hidden'] = map(lambda i: opts['hidden'][i], n_hidden_ind)
    mparams['drates'] = map(lambda x:x[1], sorted((k,v) for k, v in best.iteritems() if k.startswith('d')))
    return mparams

if __name__ == '__main__':
    main()


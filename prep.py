import tarfile
import random
import numpy as np

# [(1537882, 600), (10000, 600), (10000, 600)]
import socket
hostname = socket.gethostname().split('.')[0]

if hostname in ['balina','ural','altay']:
    DATA_DIR = '/ai/home/okuru13/data'
else:
    DATA_DIR = '/mnt/kufs/scratch/okuru13/data'


def get_dset(dname):
    return np.loadtxt('{}/1e9.p{}'.format(DATA_DIR,dname))

def get_dset_tar(dname):
    with tarfile.open('{}/1e9data.tgz'.format(DATA_DIR), 'r:gz') as src:
        devfile = src.getmember('1e9.p'+dname)
        devf = src.extractfile(devfile)
        return np.loadtxt(devf)

def prep_toy():
    trn, dev, tst = map(get_dset, ('trn','dev','tst'))
    trn_row_ind = np.random.permutation(np.arange(trn.shape[0]))[:5000]
    dev_row_ind = np.random.permutation(np.arange(dev.shape[0]))[:1000]
    tst_row_ind = np.random.permutation(np.arange(tst.shape[0]))[:1000]
    np.savez_compressed('toy.npz',
            trn=trn[trn_row_ind,:],
            dev=dev[dev_row_ind,:],
            tst=tst[tst_row_ind,:]
            )

if __name__ == '__main__':
    dev = get_dset('dev')
    Y = dev[:,:200]
    print np.mean(np.sqrt(np.sum(Y**2,axis=1)))

import tarfile
import random
import numpy as np

# [(1537882, 600), (10000, 600), (10000, 600)]

def get_dset(dname):
    with tarfile.open('/ai/tmp/1e9data.tgz', 'r:gz') as src:
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
    prep_toy()

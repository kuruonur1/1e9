import lasagne, theano, numpy as np, logging
from theano import tensor as T
from batch_norm import batch_norm

class DNN(object):
    param_names = ['activation','n_hidden','drates','opt','lr','norm','n_batch','bnorm']

    def __init__(self, nf, nout, kwargs):

        for pname in DNN.param_names:
            setattr(self, pname, kwargs[pname])

        if self.activation == 'elu':
            nonlin = lambda x: T.switch(x >= 0, x, T.exp(x) - 1)
        else:
            self.activation = 'rectify' if self.activation == 'relu' else self.activation
            nonlin = getattr(lasagne.nonlinearities, self.activation)
        self.opt = getattr(lasagne.updates, self.opt)

        l_in = lasagne.layers.InputLayer(shape=(None, nf))
        cur_layer = batch_norm(l_in) if self.bnorm else l_in
        cur_layer = lasagne.layers.DropoutLayer(cur_layer, p=self.drates[0]) if self.drates[0] > 0 else cur_layer

        self.layers = [cur_layer]
        for n_hidden, drate in zip(self.n_hidden,self.drates[1:]):
            l_betw = lasagne.layers.DenseLayer(self.layers[-1], num_units=n_hidden, nonlinearity=nonlin)
            cur_layer = batch_norm(l_betw) if self.bnorm else l_betw
            cur_layer = lasagne.layers.DropoutLayer(cur_layer, p=drate) if drate > 0 else cur_layer
            self.layers.append(cur_layer)

        l_out = lasagne.layers.DenseLayer(self.layers[-1], num_units=nout, nonlinearity=None)

        target_output = T.matrix('target_output')


        # cost_train = T.mean(lasagne.objectives.squared_error(lasagne.layers.get_output(l_out, deterministic=False), target_output))
        cost_train = T.mean(T.sum(lasagne.objectives.squared_error(lasagne.layers.get_output(l_out, deterministic=False), target_output), axis=1) / 2)
        cost_eval = T.mean(T.sum(lasagne.objectives.squared_error(lasagne.layers.get_output(l_out, deterministic=True), target_output), axis=1) / 2)
        # cost_eval = T.mean((lasagne.layers.get_output(l_out, deterministic=True)-target_output)**2)

        all_params = lasagne.layers.get_all_params(l_out, trainable=True)
        all_grads = T.grad(cost_train, all_params)

        all_grads, total_norm = lasagne.updates.total_norm_constraint(all_grads, self.norm, return_norm=True)
        # all_grads = [T.switch(T.or_(T.isnan(total_norm), T.isinf(total_norm)), p*0.01 , g) for g,p in zip(all_grads, all_params)]
        updates = self.opt(all_grads, all_params, self.lr)

        self.train_model = theano.function(
                inputs=[l_in.input_var, target_output],
                outputs=cost_train,
                updates=updates,
                allow_input_downcast=True)
        self.predict_model = theano.function(
                inputs=[l_in.input_var, target_output],
                outputs=[cost_eval,lasagne.layers.get_output(l_out, deterministic=True)],
                allow_input_downcast=True)

    def train(self, X, Y):
        batch_size = self.n_batch
        offsets = range(0, X.shape[0], batch_size)
        costs = [self.train_model(X[o:o+batch_size,:], Y[o:o+batch_size,:]) for o in offsets]
        return sum(costs)/len(costs)
    
    def predict(self, X, Y):
        batch_size = self.n_batch
        offsets = range(0, X.shape[0], batch_size)
        costsNpreds = [self.predict_model(X[o:o+batch_size,:], Y[o:o+batch_size,:]) for o in offsets]
        costs = map(lambda x:x[0], costsNpreds)
        preds = np.vstack(map(lambda x:x[1], costsNpreds))
        return sum(costs)/len(costs), preds

if __name__ == '__main__':
    params = {'activation':'tanh','n_hidden':[100],'drates':[0,0],'opt':'adam','lr':0.1,'norm':5}
    dnn = DNN(20,10,params)
    pass

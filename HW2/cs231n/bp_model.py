import numpy as np
import matplotlib.pyplot as plt

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class BPNET(object):
  def __init__(self, weight_scale=1e-3, bias_scale=0, input_shape=(3, 32, 32), num_classes=2, num_filters=[32,64,128], num_hidden=[1000], filter_size=5, pool_cita=None):
    C, H, W = input_shape
    assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size
    self.model = {}
    self.pool_cita = pool_cita
    self.conv_ctr = len(num_filters)
    self.affn_ctr = len(num_hidden)
    
    in_c = C 
    
    for idx,out_channel in enumerate(num_filters):
      self.model["convW" + str(idx)] = weight_scale * np.random.randn(out_channel, in_c, filter_size, filter_size)
      self.model["convb" + str(idx)] = bias_scale * np.random.randn(out_channel)
      in_c = out_channel


    for idx,out_channel in enumerate(num_hidden):
        if idx == 0:
            in_c = in_c * H * W

        self.model['fcW' + str(idx)] = weight_scale * np.random.randn(in_c, out_channel)
        self.model['fcb' + str(idx)] = bias_scale * np.random.randn(out_channel)
        in_c = out_channel
    

    self.model['outW'] = weight_scale * np.random.randn(in_c, num_classes)
    self.model['outb'] = bias_scale * np.random.randn(num_classes)

  def loss(self, X, y=None, reg=0.0):
    X = np.expand_dims(np.expand_dims(X, axis=1),axis=1)
    
    N, C, H, W = X.shape

    Wlist = []

    conv_filter_height, conv_filter_width = self.model["convW0"].shape[2:]
    assert conv_filter_height == conv_filter_width, 'Conv filter must be square'
    assert conv_filter_height % 2 == 1, 'Conv filter height must be odd'
    assert conv_filter_width % 2 == 1, 'Conv filter width must be odd'
    conv_param = {'stride': 1, 'pad': (conv_filter_height - 1) // 2}
    pool_param = {'pool_height': 1, 'pool_width': 1, 'stride': 1}


    cacheConv = []
    cacheAffn = []
    feature = X;

    for i in range(self.conv_ctr):
        Ws = self.model["convW" + str(i)]
        bs = self.model["convb" + str(i)]
        if self.pool_cita == i:
          feature, cache = conv_relu_pool_forward(feature, Ws, bs, conv_param, pool_param)
          cacheConv.append(cache)
          Wlist.append(Ws)
        else:
          feature, cache = conv_relu_forward(feature, Ws, bs, conv_param)
          cacheConv.append(cache)
          Wlist.append(Ws)
    
    for i in range(self.affn_ctr):
        Ws = self.model["fcW" + str(i)]
        bs = self.model["fcb" + str(i)]
        feature, cache = affine_forward(feature, Ws, bs)
        cacheAffn.append(cache)
        Wlist.append(Ws)

    fcW,fcb = self.model["outW"], self.model["outb"]
    Wlist.append(fcW)
    scores, top_cache = affine_forward(feature, fcW, fcb)

    if y is None:
      return scores

    # Compute the backward pass                                                                                                                                                                                                                                           
    data_loss, dscores = self.sqerror(scores, y, reg)

    # Compute the gradients using a backward pass
    grads = {}
    da, dW, db = affine_backward(dscores, top_cache)
    grads["outW"] = dW
    grads["outb"] = db

    idx = len(cacheAffn) -1
    for caches in cacheAffn[::-1]:
        da, dW, db = affine_backward(da, caches)
        grads["fcW" + str(idx)] = dW + reg * self.model["fcW" + str(idx)]
        grads["fcb" + str(idx)] = db + reg * self.model["fcb" + str(idx)]
        idx = idx - 1

    idx = len(cacheConv) -1
    for caches in cacheConv[::-1]:
        if self.pool_cita == idx:
          da, dW, db = conv_relu_pool_backward(da, caches)
          grads["convW" + str(idx)] = dW + reg * self.model["convW" + str(idx)]
          grads["convb" + str(idx)] = db + reg * self.model["convb" + str(idx)]
        else:
          da, dW, db = conv_relu_backward(da, caches)
          grads["convW" + str(idx)] = dW + reg * self.model["convW" + str(idx)]
          grads["convb" + str(idx)] = db + reg * self.model["convb" + str(idx)]                                                                                                 
        
        idx = idx - 1

    reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in Wlist)

    loss = data_loss + reg_loss
    
    return loss, grads

  def train(self, X, y, X_val, y_val,learning_rate=1e-3, learning_rate_decay=0.95, reg=1e-5, num_iters=100, batch_size=200, verbose=False, ev_step=20):
    print("BRO")
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)

    # Use SGD to optimize the parameters in self.self.model
    loss_history = []
    train_err_history = []
    val_err_history = []
    self.step_cache = {}

    for it in range(num_iters):
      rand_vals = np.random.choice(num_train,batch_size)
      X_batch,y_batch = X[rand_vals],y[rand_vals]

      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      for p in self.model:
        # i only used adam here
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        lr = learning_rate
        
        if not p in self.step_cache:
          state = {}
          state["step"]    = 0
          state["exp_avg"] = np.zeros_like(grads[p])
          state["exp_avg_sq"] = np.zeros_like(grads[p])
          self.step_cache[p] = state

        self.step_cache[p]["step"] += 1
        exp_avg    = self.step_cache[p]["exp_avg"]
        exp_avg_sq = self.step_cache[p]["exp_avg_sq"]
        step       = self.step_cache[p]["step"]

        exp_avg      = beta1 * exp_avg + (1-beta1) * grads[p]
        exp_avg_sq   = beta2 * exp_avg_sq + (1-beta2)*(grads[p]**2)
        
        denom     = np.sqrt(exp_avg_sq) + epsilon
        
        bias_cor1    = 1-beta1**step
        bias_cor2    = 1-beta2**step

        step_size = lr * np.sqrt(bias_cor2) / bias_cor1
    
        dx = -( step_size * (exp_avg / denom))
        self.step_cache[p]["exp_avg"]    = exp_avg   
        self.step_cache[p]["exp_avg_sq"] = exp_avg_sq

        self.model[p] -= dx

      if verbose and it % ev_step == 0:
        print ('iteration %d / %d: loss %f' % (it, num_iters, loss))

      if it % iterations_per_epoch == 0:

        train_err = np.sum(np.square(self.predict(X_batch) - y_batch), axis=1).mean()
        val_err   = np.sum(np.square(self.predict(X_val) - y_val), axis=1).mean()
        train_err_history.append(train_err)
        val_err_history.append(val_err)

        print("Val error : {:.3f} , Train error : {:.3f} ".format(train_err,val_err))
        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_err_history': train_err_history,
      'val_err_history': val_err_history,
    }

  def predict(self, X):

    y_pred = self.loss(X)
    return y_pred
  
  def wrapit(self,data):
    return {"data":data}

  def sqerror(self, y, wrapper ,reg):
    inter = wrapper - y
    loss  = np.sum( inter**2 /(2*y.shape[0]) )
    #loss += reg * np.sum([np.sum(p**2) for p in self.model.values()] ) / 2
    return loss, inter
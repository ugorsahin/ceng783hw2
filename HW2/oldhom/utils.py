import _pickle as cPickle
import gzip
import scipy.io as sio # you may use scipy for loading your data
from math import sqrt, ceil
import numpy as np
from random import randrange
import os
import h5py
import scipy
import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array

def load_mnist(path = '../data/mnist.pkl.gz'):
    "Load the MNIST dataset and return training, validation and testing data separately"
    f = gzip.open(path, 'rb')
    training_data, validation_data, test_data = cPickle.load(f,encoding='latin1')
    f.close()

    X_train, y_train = training_data[0], training_data[1]
    X_val, y_val = validation_data[0], validation_data[1]
    X_test, y_test = test_data[0], test_data[1]

    return X_train, y_train, X_val, y_val, X_test, y_test
    
    
def eval_numerical_gradient(f, x, verbose=False):
  """ 
  a naive implementation of numerical gradient of f at x 
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """

  fx = f(x) # evaluate function value at original point
  grad = np.zeros(x.shape)
  h = 0.00001

  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    x[ix] += h # increment by h
    fxh = f(x) # evalute f(x + h)
    x[ix] -= h # restore to previous value (very important!)

    # compute the partial derivative
    grad[ix] = (fxh - fx) / h # the slope
    it.iternext() # step to next dimension

  return grad

def grad_check_sparse(f, x, analytic_grad, num_checks):
  """
  sample a few random elements and only return numerical
  in this dimensions.
  """
  h = 1e-5

  x.shape
  for i in range(num_checks):
    ix = tuple([randrange(m) for m in x.shape])

    x[ix] += h # increment by h
    fxph = f(x) # evaluate f(x + h)
    x[ix] -= 2 * h # increment by h
    fxmh = f(x) # evaluate f(x - h)
    x[ix] += h # reset

    grad_numerical = (fxph - fxmh) / (2 * h)
    grad_analytic = analytic_grad[ix]
    rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
    print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))
    
def visualize_grid(Xs, ubound=255.0, padding=1):
  """
  Reshape a 3D tensor of image data to a grid for easy visualization.

  Inputs:
  - Xs: Data of shape (D, D, H)
  - ubound: Output grid will have values scaled to the range [0, ubound]
  - padding: The number of blank pixels between elements of the grid
  """
  (D, D, H) = Xs.shape
  grid_size = int(ceil(sqrt(H)))
  grid_height = D * grid_size + padding * (grid_size - 1)
  grid_width = D * grid_size + padding * (grid_size - 1)
  grid = np.zeros((grid_height, grid_width))
  next_idx = 0
  y0, y1 = 0, D
  for y in range(grid_size):
    x0, x1 = 0, D
    for x in range(grid_size):
      if next_idx < H:
        img = Xs[:, :, next_idx]
        low, high = np.min(img), np.max(img)
        grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
        # grid[y0:y1, x0:x1] = Xs[next_idx]
        next_idx += 1
      x0 += D + padding
      x1 += D + padding
    y0 += H + padding
    y1 += H + padding
  # grid_max = np.max(grid)
  # grid_min = np.min(grid)
  # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
  return grid



def load_BP_dataset(filename):
  """Load your 'PPG to blood pressure' dataset"""
  # TODO: Fill this function so that your version of the data is loaded from a file into vectors
  length = 1000
  file = h5py.File(filename)
  keys = file["Part_1"]
  X    = np.zeros((3000,length))
  Y    = np.zeros((3000,2))
    
  num_ins = keys.shape[0]
  for i in range(keys.shape[0]):
    t1 = int((i+1) * 10/ num_ins)
    t2 = 10 -t1
    print("\rLoading : |" + "#" * t1 + "-"*t2 + "|" ,end="")
    data  = np.array(file[keys[i][0]])
    X[i]  = data.T[0,:length]
    maks,mins = peakdet(data.T[1,:length],4)
    
    
    if maks.shape[0] == 0 or mins.shape[0] == 0:
      continue
    max_val = maks.T[1]
    min_val = mins.T[1]
    Y[i][0] = np.mean(max_val)
    Y[i][1] = np.mean(min_val)
  print()
  return X,Y

def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)
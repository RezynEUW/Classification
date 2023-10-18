from typing import Iterable, Callable
import numpy as np

L = 10

def logisticFunc( x : np.ndarray, out : np.ndarray=None):
    if out is None:
        ret = np.zeros( x.shape, dtype=x.dtype)
        np.exp( -x, out=ret)
        np.add( ret, 1, out=ret)
        np.divide( L, ret, out=ret)
        return ret
    else:
        np.exp( -x, out=out)
        np.add( out, 1, out=out)
        np.divide( L, out, out=out)
    

def dLogisticFunc( x : np.ndarray, out : np.ndarray=None):
    if out is None:
        ret = np.zeros(x.shape, dtype=x.dtype)
        logisticFunc( x, out=ret)
        np.multiply( ret, logisticFunc( 1 - x), out=ret)
        return ret
    else:
        logisticFunc( x, out=out)
        np.multiply( out, logisticFunc( 1 - x), out=out)

class Layer:
    size : int
    a : np.ndarray
    W : np.ndarray
class Weights:
    pass

class Weights(np.ndarray):
    def __new__( cls, layer1Size : Layer, layer2Size : Layer):
        return np.ndarray.__new__( cls, shape=(layer2Size, layer1Size), dtype=np.float32)
    def __init__( self, layer1Size : Layer, layer2Size : Layer):
        # self.fill( 1.)
        pass

class Layer:
    size : int
    a : np.ndarray
    z : np.ndarray
    b : np.ndarray
    W : np.ndarray
    def __init__( self, size):
        self.size = size
        self.a = np.zeros((self.size), dtype=np.float32)
        self.z = np.zeros((self.size), dtype=np.float32)
        self.b = None
        self.W = None

class aNN:
    layers : list[Layer]
    labels : np.ndarray
    # func : Callable
    costs : Iterable[np.ndarray]

    def __init__( self, layers : tuple[int], labels : np.ndarray, *args, dataSize=None, func=lambda x:x, **kwargs):
        if dataSize is None:
            self.layers = [Layer( size) for size in (*layers, len( labels))]
            self.costs = np.zeros( (len( layers)-1 + len( labels)), dtype=np.float32)
        else:
            self.layers = [Layer( size) for size in (dataSize, *layers, len( labels))]
            self.costs = np.zeros( (len( layers) + len( labels)), dtype=np.float32)
        self.labels = labels
        # self.func = func
        for i in range( len( self.layers)-1):
            self.layers[i].W = Weights( self.layers[i].size, self.layers[i+1].size)
            self.layers[i].b = np.zeros(( self.layers[i+1].size))
    
    def loadSample( self, sample : np.ndarray):
        self.layers[0].a = sample.flatten()

    def propagate( self, sample : np.ndarray):
        self.loadSample( sample)
        for i in range( len( self.layers)-1):
            self.layers[i+1].z = self.layers[i].W.dot( self.layers[i].a) + self.layers[i].b
            logisticFunc( self.layers[i+1].z, out=self.layers[i+1].a)

    def predict( self, samples : np.ndarray, out=None):
        if out is None:
            results = np.zeros((range(samples),), dtype=self.labels.dtype)
        else:
            results = out
        
        for i, sample in enumerate( samples):
            self.propagate( sample)
            results[i] = self.labels[np.argmax( self.layers[-1].a)]
        
        if out is None:
            return results
        else:
            return None
    
    def train( self):
        """
        Not implemented in the parent class. Each implementation of the aNN has
        its own algorithm for adjusting the weights.
        """
        pass

class backPropNN(aNN):
    backPropLayers : list[np.ndarray]
    def __init__( self, layers : tuple[int], *args, **kwargs):
        super().__init__( layers, *args, **kwargs)
        self.backPropLayers = [Weights( *layer.W.shape) for layer in self.layers[-2::-1]]
    
    def backPropagate( self, samples : np.ndarray, results : np.ndarray):
        # Zero out the weight error matrices
        dW = [np.zeros( layer.W.shape) for layer in self.layers[:-1]]
        db = [np.zeros( layer.b.shape) for layer in self.layers[:-1]]
        for sample, answer in zip( samples, results):
            self.propagate( sample)

            error = self.layers[-1].a - answer
            for i in range( 1, len( self.layers)):
                # Derivative of the cost depending on the node value
                dCost = np.sum( 2 * error)
                # Derivative of the logistic function at the given node value
                dLog = dLogisticFunc( self.layers[-i].z)
                # Change in weight
                dW[-i] += (dLog * dCost).reshape( 1, len( dLog)).T.dot( self.layers[-i-1].a.reshape( 1, len( self.layers[-i-1].a)))
                # Change in bias
                db[-i] += dLog * dCost
                
                # Change in node value (The error for the next iteration)
                if i < len( self.layers) - 1:
                    error = self.layers[-i-1].W.T.dot( dLog * dCost)
        
        # Change the sum of changes to a mean of changes and apply to the
        # network
        for i in range( len( dW)):
            self.layers[i].W += dW[i] / len( samples)
            self.layers[i].b += db[i] / len( samples)
    
    def train( self, sampleData : np.ndarray, sampleLabels : np.ndarray):
        results = np.zeros((len( sampleLabels), len( self.labels)), dtype=np.float32)
        results[:,:] = sampleLabels.reshape( len( sampleLabels), 1)
        results = np.equal( results, self.labels)
        self.backPropagate( sampleData, results)
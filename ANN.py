from typing import Iterable
import numpy as np


class Layer:
    size : int
    a : np.ndarray
    W : np.ndarray
    def __init__( self, size):
        self.size = size
        self.a = np.zeros((self.size), dtype=np.float32)
        self.W = None

class aNN:
    layers : list[Layer]
    def __init__( self, layers : tuple[int], labels, *args, **kwargs):
        self.layers = [Layer( size) for size in layers]
        self.labels = labels
        for i in range( len( self.layers)-1):
            self.layers[i].W = np.zeros((self.layers[i].size, self.layers[i+1].size), dtype=np.float32)
            self.layers[i].W.fill( 1.)
    
    def loadSample( self, sample : np.ndarray):
        self.layers[0].a = sample.flatten()

    def propagate( self, sample : np.ndarray):
        self.loadSample( sample)
        for i in range( len( self.layers)-1):
            self.layers[i+1].a = self.layers[i].W.dot( self.layers[i].a)

    def predict( self, samples : np.ndarray):
        results = []
        for sample in samples:
            self.propagate( sample)
            results.append( self.labels[np.argmax( self.layers[-1].a)])
        return np.array( results)

class backPropNN(aNN):
    backPropLayers : list[Layer]
    def __init__( self, layers : tuple[int], *args, **kwargs):
        super().__init__( self, layers, *args, **kwargs)
        self.backPropLayers = [Layer( size) for size in layers[::-1]]
    
    def backPropagate( self, samples : np.ndarray, results : np.ndarray):
        for sample, answer in zip( samples, results):
            self.propagate( sample)
            answer - self.layers[-1].a
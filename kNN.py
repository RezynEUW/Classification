import numpy as np
from typing import Any, Callable, Union

class kNN:
    process : np.ndarray
    trainingData : np.ndarray
    processed : np.ndarray
    preProcessFunc : tuple[Callable, list, dict]
    labels : np.ndarray
    k : Union[int,np.int32]

    def __init__( self, trainingData : np.ndarray, labels : np.ndarray, k : Union[int,np.int32]=1, func : Callable=np.mean):
        self.trainingData = trainingData
        self.processed = None
        self.preProcessFunc = (lambda x: x, [], {})
        self.labels = labels
        self.setK( k)
    
    def preProcess( self, func : Callable, *args, **kwargs):
        self.preProcessFunc = (func, args, kwargs)
        self.processed = func( self.trainingData, *args, **kwargs)

        # Preallocate the space for error values
        self.error = np.zeros( self.processed.shape, dtype=self.processed.dtype)
        
        # Preallocate the space for scoring vectors
        self.scores = np.zeros( (self.processed.shape[0]), dtype=self.processed.dtype)

    def setK( self, k):
        self.k = k

    def train( self, trainingData : np.ndarray, labels : np.ndarray):
        self.trainingData = trainingData
        self.labels = labels

        self.processed = None
        self.preProcessFunc = (lambda x: x, [], {})

    def predict( self, samples : np.ndarray) -> Union[np.ndarray,Any]:
        # Check if there are multiple samples or just one
        if samples.ndim == self.trainingData.ndim:
            _samples = samples
        elif samples.ndim == self.trainingData.ndim - 1:
            # Add sample dimension for compatibility with the implementation
            _samples = samples.reshape((1, *samples.shape))
        _samples = self.preProcessFunc[0]( _samples, *self.preProcessFunc[1], **self.preProcessFunc[2])

        nSamples = _samples.shape[0]

        # Check if data has not been pre-processed
        if self.processed is None:
            self.processed = self.trainingData
            
            # Preallocate the space for error values
            self.error = np.zeros( self.processed.shape, dtype=np.float32)
            
            # Preallocate the space for scoring vectors
            self.scores = np.zeros( (self.processed.shape[0]), dtype=np.float32)

        results = np.zeros((nSamples, self.labels.size), dtype=self.labels.dtype)

        labelLookup = np.unique( self.labels)
        labelIndex  = {l:i for i, l in enumerate( labelLookup)}
        for i in range( nSamples):
            
            np.subtract( _samples[i], self.processed, out=self.error)
            np.square( self.error, out=self.error)
            np.sum( self.error, axis=(1,2) ,out=self.scores)

            m = np.amax( self.scores)
            for k in range( self.k):
                # The lowest scoring training image has a label assigned to it
                # Increment the result counter in the position associated with
                # the given label
                results[i,labelIndex[self.labels[np.argmin( self.scores)]]] += 1

                # Set the score to something that will give it lowest priority
                self.scores[np.argmin( self.scores)] = m
        
        return labelLookup[np.argmax( results, axis=1)]
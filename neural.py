import numpy as np


class NeuralNetwork:
    def __init__( self, layers : tuple, labels : tuple):
        self.labels = labels
        self.shape = layers
        self.layers = [np.zeros(n, dtype=np.float32) for n in layers]
        self.dependencies = [np.zeros((layers[i+1],layers[i]), dtype=np.float32) for i in range( len(layers)-1)]
        self.scorings = [np.zeros((2,layers[i+1],layers[i]), dtype=np.int32) for i in range( len(layers)-1)]

        for i in range( len( self.dependencies)):
            self.dependencies[i][::] = 0.5
    
    def sharpen( self, data):
        
        for n in range( len( self.layers)):
            for i in range( len( self.dependencies[n].shape[0])):
                for j in range( len( self.dependencies[n].shape[1])):
                    old = self.dependencies[n][i,j]
                    self.dependencies[n][i,j] = old + 0.1
                    if self.apply( data["images"][n]) == data["labels"][n]:
                        self.scorings[n][0,i,j] += 1
                    
                    self.dependencies[n][i,j] = old - 0.1
                    if self.apply( data["images"][n]) == data["labels"][n]:
                        self.scorings[n][1,i,j] += 1
        
        biggies = [(n, np.argmax( self.scorings[n])) for n in range( len( self.layers))]

        bestChange = max( biggies, key=lambda x: self.scorings[x[0]][x[1]])
        if bestChange[1][0] == 0:
            self.dependencies[bestChange[0]][bestChange[1][1], bestChange[1][2]] += 0.1
        else:
            self.dependencies[bestChange[0]][bestChange[1][1], bestChange[1][2]] -= 0.1

    def apply( self, data):
        self.layers[0] = data
        for n in range( len( self.shape)-1):
            self.layers[n+1] = self.dependencies[n].dot( self.layers[n])
        
        return self.labels[ np.argmax( self.layers[-1])]
    
    def validate( self, data):
        score = 0
        for i in range( len( data)):
            if data["labels"][i] == self.apply( data["images"][i]):
                score += 1
        return score / len( data)
    
    def train( self, trainingData, validationData=None, n=1):
        for i in range( n):
            self.sharpen( trainingData)
            print( "i: {:<5d}, PT = {:.3f}, PV = {:.3f}".format( i, self.validate( trainingData), self.validate( validationData) if validationData is not None else 0.0))
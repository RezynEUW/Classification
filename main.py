
from idxImporter import interpretIDX
from kNN import kNN
from aaNN import backPropNN
import os
import numpy as np

from preProcessing import boxBlur, edgeRecognition

PATH_TO_TRAINING_IMAGES = os.path.sep.join( ["trainingData", "train-images.idx3-ubyte"])
PATH_TO_TRAINING_LABELS = os.path.sep.join( ["trainingData", "train-labels.idx1-ubyte"])

PATH_TO_VALIDATION_IMAGES = os.path.sep.join( ["validationData", "t10k-images.idx3-ubyte"])
PATH_TO_VALIDATION_LABELS = os.path.sep.join( ["validationData", "t10k-images.idx1-ubyte"])

if __name__ == "__main__":
    from timeit import default_timer as timer
    
    start = timer()
    print( "Loading data.. ", end="")

    trainingRawDataImages = open( PATH_TO_TRAINING_IMAGES, "rb")
    trainingRawDataLabels = open( PATH_TO_TRAINING_LABELS, "rb")

    [TDIdict] = interpretIDX( trainingRawDataImages)
    [TDLdict] = interpretIDX( trainingRawDataLabels)

    trainingDataImages = np.divide( TDIdict["dataArray"], 255, dtype=np.float32)
    trainingDataLabels = TDLdict["dataArray"]

    print( "{:.2f} s.".format( timer()-start))

    labels = np.sort( np.unique( trainingDataLabels))

    training   = (trainingDataImages[::10], trainingDataLabels[::10])
    validation = (trainingDataImages[1::50], trainingDataLabels[1::50])

    
    tryKNN = True

    if tryKNN:
        start = timer()
        print( "Setting up kNN.. ", end="")

        # preProcess images
        if True:
            # training = (boxBlur(training[0], 1), training[1])
            # validation = (boxBlur(validation[0], 1), validation[1])
            
            training = (edgeRecognition(training[0]), training[1])
            validation = (edgeRecognition(validation[0]), validation[1])

        k = 4
        
        kNNmodel = kNN( *training, k)

        # kNNmodel.preProcess( boxBlur, 2)
        
        print( "{:.2f} s.".format( timer()-start))
        start = timer()
        print( "Predicting.. ", end="")

        results = kNNmodel.predict( validation[0])

        print( "{:.2f} s.".format( timer()-start))
        
        print( "Correctly guessed labels: {} ({:.5f} %)".format( np.sum( validation[1] == results), 100 * np.mean( validation[1] == results)))
    
    tryANN = False

    if tryANN:
        start = timer()
        print( "Setting up aNN.. ", end="")

        # preProcess images
        if False:
            training = (boxBlur(training[0], 2), training[1])
            validation = (boxBlur(validation[0], 2), validation[1])
        
        layers = (16, 16)
        imageSize = training[0].shape[1] * training[0].shape[2]
        
        aNNmodel = backPropNN( layers, labels, dataSize=imageSize)

        strides = 7
        trainingSets = [(training[0][i::strides], training[1][i::strides]) for i in range( strides)]
        
        i = 0
        results = np.zeros( (len(validation[0]),), aNNmodel.labels.dtype)
        while np.mean( validation[1] == results) < 0.977:
            aNNmodel.train( *trainingSets[i % strides])
            aNNmodel.predict( validation[0], out=results)
            print( "Correctly guessed labels: {} ({:.5f} %)".format( np.sum( validation[1] == results), 100 * np.mean( validation[1] == results)))
            print( np.mean( aNNmodel.layers[1].W), aNNmodel.layers[1].W.std())
            i += 1
        
        print( "{:.2f} s.".format( timer()-start))
        start = timer()
        print( "Predicting.. ", end="")

        results = aNNmodel.predict( validation[0])

        print( "{:.2f} s.".format( timer()-start))
        
        print( "Correctly guessed labels: {} ({:.5f} %)".format( np.sum( validation[1] == results), 100 * np.mean( validation[1] == results)))

    # start = timer()
    # print( "Import pyplot.. ", end="")
    # import matplotlib.pyplot as plt
    # print( "{:.2f} s.".format( timer()-start))

    # for guess, answer in zip( results, trainingDataLabels[samples]):
    #     print( "{:>4} {} {:<4}".format( guess, "=>" if guess == answer else "=/", answer))


    # plt.hist( np.mean( kNearest[1], axis=1))
    # plt.show()

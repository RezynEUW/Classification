from typing import Union, Callable
import numpy as np

ridgeMatrix = np.array([[-1,-1,-1],[-1,4,-1],[-1,-1,-1]], np.float32)
hardRidgeMatrix = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], np.float32)

def boxBlur( data : np.ndarray, boxSize : Union[int, tuple[int,int]]) -> np.ndarray:
    N, y, x = data.shape
    
    if type( boxSize) in [int, np.int_]:
        k_y = k_x = boxSize
    else:
        k_y , k_x = boxSize
    
    # Preallocate the space for the pre-processed values
    processed = np.zeros((N, y-2*k_y, x-2*k_x), dtype=np.float32 if data.dtype is not np.float64 else np.float64)

    for y in range( processed.shape[1]):
        for x in range( processed.shape[2]):
            np.mean( data[:, y:y+2*k_y+1 , x:x+2*k_x+1 ], axis=(1,2), out=processed[:, y , x])
    
    return processed

def edgeRecognition( data : np.ndarray) -> np.ndarray:
    N, y, x = data.shape
    
    # Preallocate the space for the pre-processed values
    processed = np.zeros((N, y-2, x-2), dtype=np.float32 if data.dtype is not np.float64 else np.float64)

    for y in range( processed.shape[1]):
        for x in range( processed.shape[2]):
            np.sum(np.multiply( data[:, y:y+3 , x:x+3 ], ridgeMatrix), axis=(1,2), out=processed[:, y , x])
    
    return processed
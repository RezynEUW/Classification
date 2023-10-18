from io import TextIOWrapper
import struct
import numpy as np


IDXdataTypes = {
    b"\x08"[0]  : np.ubyte, # "unsigned byte"
    b"\x09"[0]  : np.byte, # "signed byte"
    b"\x0B"[0]  : np.short, # "short (2 bytes)"
    b"\x0C"[0]  : np.int32, # "int (4 bytes)"
    b"\x0D"[0]  : np.float32, # "float (4 bytes)"
    b"\x0E"[0]  : np.double # "double (8 bytes)"
}

letterTypes = {
    np.ubyte    : "B",
    np.byte     : "b",
    np.short    : "h",
    np.int32    : "l",
    np.float32  : "f",
    np.double   : "d"
}

byteSizes = {
    np.ubyte    : 1,
    np.byte     : 1,
    np.short    : 2,
    np.int32    : 4,
    np.float32  : 4,
    np.double   : 8
}

def interpretIDX( rawData : TextIOWrapper) -> list[dict]:
    
    data = []

    while True:
        try:
            entryData = {}
            
            null, null, dataType, entryData["dataDimension"] = struct.unpack( ">4B", rawData.read(4))
            
            entryData["dataType"] = IDXdataTypes[dataType]

            entryData["dimensionSizes"] = struct.unpack( ">{}l".format( entryData["dataDimension"]), rawData.read( 4 * entryData["dataDimension"]))
            
            entryData["entrySize"] = 1
            for i in range( entryData["dataDimension"]):
                entryData["entrySize"] *= entryData["dimensionSizes"][i]
            entryData["dataArray"] = np.array( struct.unpack( ">{}{}".format( entryData["entrySize"], letterTypes[entryData["dataType"]]), rawData.read( entryData["entrySize"] * byteSizes[entryData["dataType"]])), dtype=entryData["dataType"]).reshape( entryData["dimensionSizes"])

            data.append( entryData)
            
        except Exception as e:
            break
    
    return data
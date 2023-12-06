# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class DimensionMetadata(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsDimensionMetadata(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DimensionMetadata()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def DimensionMetadataBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # DimensionMetadata
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # DimensionMetadata
    def Format(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # DimensionMetadata
    def DenseSize(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # DimensionMetadata
    def ArraySegmentsType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    # DimensionMetadata
    def ArraySegments(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            from flatbuffers.table import Table
            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None

    # DimensionMetadata
    def ArrayIndicesType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    # DimensionMetadata
    def ArrayIndices(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            from flatbuffers.table import Table
            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None

def DimensionMetadataStart(builder): builder.StartObject(6)
def DimensionMetadataAddFormat(builder, format): builder.PrependInt8Slot(0, format, 0)
def DimensionMetadataAddDenseSize(builder, denseSize): builder.PrependInt32Slot(1, denseSize, 0)
def DimensionMetadataAddArraySegmentsType(builder, arraySegmentsType): builder.PrependUint8Slot(2, arraySegmentsType, 0)
def DimensionMetadataAddArraySegments(builder, arraySegments): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(arraySegments), 0)
def DimensionMetadataAddArrayIndicesType(builder, arrayIndicesType): builder.PrependUint8Slot(4, arrayIndicesType, 0)
def DimensionMetadataAddArrayIndices(builder, arrayIndices): builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(arrayIndices), 0)
def DimensionMetadataEnd(builder): return builder.EndObject()

import tflite_model.Int32Vector
import tflite_model.SparseIndexVector
import tflite_model.Uint16Vector
import tflite_model.Uint8Vector
try:
    from typing import Union
except:
    pass

class DimensionMetadataT(object):

    # DimensionMetadataT
    def __init__(self):
        self.format = 0  # type: int
        self.denseSize = 0  # type: int
        self.arraySegmentsType = 0  # type: int
        self.arraySegments = None  # type: Union[None, tflite_model.Int32Vector.Int32VectorT, tflite_model.Uint16Vector.Uint16VectorT, tflite_model.Uint8Vector.Uint8VectorT]
        self.arrayIndicesType = 0  # type: int
        self.arrayIndices = None  # type: Union[None, tflite_model.Int32Vector.Int32VectorT, tflite_model.Uint16Vector.Uint16VectorT, tflite_model.Uint8Vector.Uint8VectorT]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        dimensionMetadata = DimensionMetadata()
        dimensionMetadata.Init(buf, pos)
        return cls.InitFromObj(dimensionMetadata)

    @classmethod
    def InitFromObj(cls, dimensionMetadata):
        x = DimensionMetadataT()
        x._UnPack(dimensionMetadata)
        return x

    # DimensionMetadataT
    def _UnPack(self, dimensionMetadata):
        if dimensionMetadata is None:
            return
        self.format = dimensionMetadata.Format()
        self.denseSize = dimensionMetadata.DenseSize()
        self.arraySegmentsType = dimensionMetadata.ArraySegmentsType()
        self.arraySegments = tflite_model.SparseIndexVector.SparseIndexVectorCreator(self.arraySegmentsType, dimensionMetadata.ArraySegments())
        self.arrayIndicesType = dimensionMetadata.ArrayIndicesType()
        self.arrayIndices = tflite_model.SparseIndexVector.SparseIndexVectorCreator(self.arrayIndicesType, dimensionMetadata.ArrayIndices())

    # DimensionMetadataT
    def Pack(self, builder):
        if self.arraySegments is not None:
            arraySegments = self.arraySegments.Pack(builder)
        if self.arrayIndices is not None:
            arrayIndices = self.arrayIndices.Pack(builder)
        DimensionMetadataStart(builder)
        DimensionMetadataAddFormat(builder, self.format)
        DimensionMetadataAddDenseSize(builder, self.denseSize)
        DimensionMetadataAddArraySegmentsType(builder, self.arraySegmentsType)
        if self.arraySegments is not None:
            DimensionMetadataAddArraySegments(builder, arraySegments)
        DimensionMetadataAddArrayIndicesType(builder, self.arrayIndicesType)
        if self.arrayIndices is not None:
            DimensionMetadataAddArrayIndices(builder, arrayIndices)
        dimensionMetadata = DimensionMetadataEnd(builder)
        return dimensionMetadata

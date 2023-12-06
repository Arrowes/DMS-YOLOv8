# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class FakeQuantOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsFakeQuantOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FakeQuantOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def FakeQuantOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # FakeQuantOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # FakeQuantOptions
    def Min(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    # FakeQuantOptions
    def Max(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    # FakeQuantOptions
    def NumBits(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # FakeQuantOptions
    def NarrowRange(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

def FakeQuantOptionsStart(builder): builder.StartObject(4)
def FakeQuantOptionsAddMin(builder, min): builder.PrependFloat32Slot(0, min, 0.0)
def FakeQuantOptionsAddMax(builder, max): builder.PrependFloat32Slot(1, max, 0.0)
def FakeQuantOptionsAddNumBits(builder, numBits): builder.PrependInt32Slot(2, numBits, 0)
def FakeQuantOptionsAddNarrowRange(builder, narrowRange): builder.PrependBoolSlot(3, narrowRange, 0)
def FakeQuantOptionsEnd(builder): return builder.EndObject()


class FakeQuantOptionsT(object):

    # FakeQuantOptionsT
    def __init__(self):
        self.min = 0.0  # type: float
        self.max = 0.0  # type: float
        self.numBits = 0  # type: int
        self.narrowRange = False  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        fakeQuantOptions = FakeQuantOptions()
        fakeQuantOptions.Init(buf, pos)
        return cls.InitFromObj(fakeQuantOptions)

    @classmethod
    def InitFromObj(cls, fakeQuantOptions):
        x = FakeQuantOptionsT()
        x._UnPack(fakeQuantOptions)
        return x

    # FakeQuantOptionsT
    def _UnPack(self, fakeQuantOptions):
        if fakeQuantOptions is None:
            return
        self.min = fakeQuantOptions.Min()
        self.max = fakeQuantOptions.Max()
        self.numBits = fakeQuantOptions.NumBits()
        self.narrowRange = fakeQuantOptions.NarrowRange()

    # FakeQuantOptionsT
    def Pack(self, builder):
        FakeQuantOptionsStart(builder)
        FakeQuantOptionsAddMin(builder, self.min)
        FakeQuantOptionsAddMax(builder, self.max)
        FakeQuantOptionsAddNumBits(builder, self.numBits)
        FakeQuantOptionsAddNarrowRange(builder, self.narrowRange)
        fakeQuantOptions = FakeQuantOptionsEnd(builder)
        return fakeQuantOptions

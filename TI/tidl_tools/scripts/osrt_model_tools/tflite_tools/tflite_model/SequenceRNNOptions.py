# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class SequenceRNNOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsSequenceRNNOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SequenceRNNOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def SequenceRNNOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # SequenceRNNOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SequenceRNNOptions
    def TimeMajor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # SequenceRNNOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # SequenceRNNOptions
    def AsymmetricQuantizeInputs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

def SequenceRNNOptionsStart(builder): builder.StartObject(3)
def SequenceRNNOptionsAddTimeMajor(builder, timeMajor): builder.PrependBoolSlot(0, timeMajor, 0)
def SequenceRNNOptionsAddFusedActivationFunction(builder, fusedActivationFunction): builder.PrependInt8Slot(1, fusedActivationFunction, 0)
def SequenceRNNOptionsAddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs): builder.PrependBoolSlot(2, asymmetricQuantizeInputs, 0)
def SequenceRNNOptionsEnd(builder): return builder.EndObject()


class SequenceRNNOptionsT(object):

    # SequenceRNNOptionsT
    def __init__(self):
        self.timeMajor = False  # type: bool
        self.fusedActivationFunction = 0  # type: int
        self.asymmetricQuantizeInputs = False  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        sequenceRNNOptions = SequenceRNNOptions()
        sequenceRNNOptions.Init(buf, pos)
        return cls.InitFromObj(sequenceRNNOptions)

    @classmethod
    def InitFromObj(cls, sequenceRNNOptions):
        x = SequenceRNNOptionsT()
        x._UnPack(sequenceRNNOptions)
        return x

    # SequenceRNNOptionsT
    def _UnPack(self, sequenceRNNOptions):
        if sequenceRNNOptions is None:
            return
        self.timeMajor = sequenceRNNOptions.TimeMajor()
        self.fusedActivationFunction = sequenceRNNOptions.FusedActivationFunction()
        self.asymmetricQuantizeInputs = sequenceRNNOptions.AsymmetricQuantizeInputs()

    # SequenceRNNOptionsT
    def Pack(self, builder):
        SequenceRNNOptionsStart(builder)
        SequenceRNNOptionsAddTimeMajor(builder, self.timeMajor)
        SequenceRNNOptionsAddFusedActivationFunction(builder, self.fusedActivationFunction)
        SequenceRNNOptionsAddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        sequenceRNNOptions = SequenceRNNOptionsEnd(builder)
        return sequenceRNNOptions

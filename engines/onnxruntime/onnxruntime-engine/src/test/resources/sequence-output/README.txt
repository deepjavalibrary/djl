Small ONNX fixtures for output ownership tests (IR version 7, opset 11).

tensors.onnx: float input [2], SequenceConstruct(input) -> sequence,
and Identity(input) -> direct. Tests independent ownership of returned
sequence tensor elements and normal tensor outputs.

maps.onnx: float input [2,2], ai.onnx.ml.ZipMap(input) -> probabilities,
with classlabels_int64s [0,1]. Tests map sequence conversion and cleanup.

Both graphs contain no learned weights or external data.

# DJL HuggingFace model converter

If you are trying to convert a complete HuggingFace (transformers) model,
you can try to use our all-in-one conversion solution to convert to Java:

Currently, this converter supports the following tasks:

- fill-mask
- question-answering
- sentence-similarity
- text-classification
- token-classification

## Install `djl_converter`

You can install `djl_converter` from djl main branch or clone the repository and install from source:

```
# install release version of djl_converter
pip install https://publish.djl.ai/djl_converter/djl_converter-0.33.0-py3-none-any.whl

# install from djl master branch
pip install "git+https://github.com/deepjavalibrary/djl.git#subdirectory=extensions/tokenizers/src/main/python"

# install djl_converter from local djl repo
git clone https://github.com/deepjavalibrary/djl.git
cd djl/extensions/tokenizers/src/main/python
python3 -m pip install -e .
```

## Convert HuggingFace model to DJL

`djl_converter` can convert HuggingFace model into TorchScript, OnnxRuntime and Rust format.

- TorchScript

```bash
# help commandline arguments
djl-convert --help

# download model from HuggingFace and convert to TorchScript
djl-convert -m deepset/bert-base-cased-squad2

# convert local HuggingFace model
djl-convert -m /opt/models/bert-base-cased-squad2

# download model from HuggingFace and convert to OnnxRuntime
djl-convert -m deepset/bert-base-cased-squad2 -f OnnxRuntime

# download model from HuggingFace and convert to Rust
djl-convert -m deepset/bert-base-cased-squad2 -f Rust
```

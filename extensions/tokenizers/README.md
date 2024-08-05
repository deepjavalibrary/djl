# NLP support with Huggingface tokenizers

This module contains the NLP support with Huggingface tokenizers implementation.

This is an implementation from [Huggingface tokenizers](https://github.com/huggingface/tokenizers) RUST API.

## Documentation

The latest javadocs can be found on [here](https://javadoc.io/doc/ai.djl.huggingface/tokenizers/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
./gradlew javadoc
```

The javadocs output is built in the `build/doc/javadoc` folder.

## Installation

You can pull the module from the central Maven repository by including the following dependency in your `pom.xml` file:

```xml

<dependency>
    <groupId>ai.djl.huggingface</groupId>
    <artifactId>tokenizers</artifactId>
    <version>0.29.0</version>
</dependency>
```

## Usage

### Use DJL HuggingFace model converter

If you are trying to convert a complete HuggingFace (transformers) model,
you can try to use our all-in-one conversion solution to convert to Java:

Currently, this converter supports the following tasks:

- fill-mask
- question-answering
- sentence-similarity
- text-classification
- token-classification

#### Install `djl-converter`

You can install `djl-converter` from djl master branch or clone the repository and install from source:

```
# install release version of djl-converter
pip install https://publish.djl.ai/djl_converter/djl_converter-0.30.0-py3-none-any.whl

# install from djl master branch
pip install "git+https://github.com/deepjavalibrary/djl.git#subdirectory=extensions/tokenizers/src/main/python"

# install djl-convert from local djl repo
git clone https://github.com/deepjavalibrary/djl.git
cd djl/extensions/tokenizers/src/main/python
python3 -m pip install -e .

# install optimum if you want to convert to OnnxRuntime
pip install optimum

# convert a single model to TorchScript, Onnxruntime or Rust
djl-convert --help

# import models as DJL Model Zoo
djl-import --help
```

#### Convert Huggingface model to torchscript

```bash
djl-convert -m deepset/bert-base-cased-squad2
```

This will find converted model in `model/bert-base-cased-squad2/` folder:

```
djl-convert -m deepset/bert-base-cased-squad2
```

#### Convert Huggingface model to OnnxRuntime

```bash
djl-convert -m deepset/bert-base-cased-squad2 -f OnnxRuntime
```

#### Convert Huggingface model to Rust

```bash
djl-convert -m deepset/bert-base-cased-squad2 -f Rust
```

#### Load converted model

Then, all you need to do, is to load this model in DJL:

```java
Criteria<QAInput, String> criteria = Criteria.builder()
        .setTypes(QAInput.class, String.class)
        .optModelPath(Paths.get("model/bert-base-cased-squad2/"))
        .optTranslatorFactory(new DeferredTranslatorFactory())
        .optProgress(new ProgressBar()).build();
```

#### Import multiple Huggingface Hub models into DJL model zoo

```
djl-import -m deepset/bert-base-cased-squad2
```

This will generate a zip file into your local djl model zoo folder structure:

```
model/nlp/question_answer/ai/djl/huggingface/pytorch/deepset/bert-base-cased-squad2/0.0.1/bert-base-cased-squad2.zip
```

### From HuggingFace AutoTokenizer

In most of the cases, you can easily use a pre-existing tokenizer in DJL:

Python

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-dot-v5")
```

Java

```java
HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance("sentence-transformers/msmarco-distilbert-dot-v5");
```

This way requires network connection to huggingface repo.
The way to determine if you can use this way is through looking into the "Files and versions"
in [HuggingFace model tab](https://huggingface.co/sentence-transformers/msmarco-distilbert-dot-v5)
and see if there is a `tokenizer.json`.

If there is a `tokenizer.json`, you can get it directly through DJL. Otherwise, use the other way below to obtain
a `tokenizer.json`.

### From HuggingFace Pipeline

If you are trying to get tokenizer from a HuggingFace pipeline,
you can use the followings to extract `tokenizer.json` file.

Python

```python
 pipeline.tokenizer.save_pretrained("./")
```

From your local directory, you will find a `tokenizer.json` file.

Java

```java
HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(Paths.get("./tokenizer.json"));
```

### From pretrained json file

Same as above step, just save your tokenizer into `tokenizer.json` (done by huggingface).

# NLP support with Huggingface tokenizers

This module contains the NLP support with Huggingface tokenizers implementation.

This is an implementation from [Huggingface tokenizers](https://github.com/huggingface/tokenizers) RUST API.

## Documentation

You can build the latest javadocs locally using the following command:

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
    <version>0.18.0</version>
</dependency>
```

## Usage

### Use DJL HuggingFace model converter (experimental)

If you are trying to convert a complete HuggingFace (transformers) model,
you can try to use our all-in-one conversion solution to convert to Java:

```bash
python3 -m pip install -r src/main/python/requirements.txt
python3 src/main/python/model_zoo_importer.py -m deepset/bert-base-cased-squad2
```

This will generate a zip file into your local folder:

```
model/nlp/question_answer/ai/djl/huggingface/pytorch/deepset/bert-base-cased-squad2/0.0.1/bert-base-cased-squad2.zip
```

Then, all you need to do, is to load this model in DJL:

```java
Criteria<QAInput, String> criteria = Criteria.builder()
    .setTypes(QAInput.class, String.class)
    .optModelPath(Paths.get("model/nlp/question_answer/ai/djl/huggingface/pytorch/deepset/bert-base-cased-squad2/0.0.1/bert-base-cased-squad2.zip"))
    .optProgress(new ProgressBar()).build();
```

Currently, this converter support:

- fill-mask
- question-answering
- sentence-similarity
- text-classification
- token-classification

### From HuggingFace AutoTokenizer
In most of the cases, you can easily use a pre-existing tokenizer in DJL:

Python

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-dot-v5")
```

Java

```java
HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance("sentence-transformers/msmarco-distilbert-dot-v5")
```

This way requires network connection to huggingface repo.
The way to determine if you can use this way is through looking into the "Files and versions" in [HuggingFace model tab](https://huggingface.co/sentence-transformers/msmarco-distilbert-dot-v5)
and see if there is a `tokenizer.json`. 

If there is a `tokenizer.json`, you can get it directly through DJL. Otherwise, use the other way below to obtain a `tokenizer.json`.

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
HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(Paths.get("./tokenizer.json"))
```

### From pretrained json file

Same as above step, just save your tokenizer into `tokenizer.json` (done by huggingface).

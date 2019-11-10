BERT QA Example
==============

In this tutorial, you'll walk through the BERT QA model trained by MXNet. 
You can provide a question and a paragraph containing the answer to the model. The model is then able to find the best answer from the answer paragraph.

Example:
```text
Q: When did BBC Japan start broadcasting?
```

Answer paragraph:
```text
BBC Japan was a general entertainment channel, which operated between December 2004 and April 2006.
It ceased operations after its Japanese distributor folded.
```
And it picked the right answer:
```text
A: December 2004
```

## Setup Guide

### Step 1: Download the model

For this tutorial, you can get the model and vocabulary by running the following commands:

```bash
cd examples/build
curl https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/BertQA/vocab.json -O
curl https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/BertQA/static_bert_qa-0002.params -O
curl https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/BertQA/static_bert_qa-symbol.json -O
```

### Step 2: Do Inference

The available arguments are as follows:

| Argument   | Comments                                 |
| ---------- | ---------------------------------------- |
| `-q`      | Question for the model |
| `-a`      | Paragraph that contains the answer |
| `-l`      | Sequence Length of the model (384 by default) |
| `-p`      | Path to the model directory |
| `-n`      |  Model name prefix |

You can type the following to run inference:

```
cd examples
./gradlew -Dmain=ai.djl.examples.inference.BertQaInference run --args="-p build/ -n static_bert_qa"
```


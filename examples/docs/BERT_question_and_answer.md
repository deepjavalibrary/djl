# BERT QA Example

In this example, you learn how to use the BERT QA model trained by MXNet. 
You can provide the model with a question and a paragraph containing an answer. The model is then able to find the best answer from the answer paragraph.
You can find the source code in [BertQaInference.java](../src/main/java/ai/djl/examples/inference/BertQaInference.java). 

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

Follow [setup](../../docs/development/setup.md) to configure your development environment.

### Run Inference

```
cd examples
./gradlew run -Dmain=ai.djl.examples.inference.BertQaInference
```

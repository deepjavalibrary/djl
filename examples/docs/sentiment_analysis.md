# Sentiment analysis example

In this example, you learn how to use the DistilBERT model trained by HuggingFace using PyTorch. 
You can provide the model with a question and a paragraph containing an answer. The model is then able to find the best answer from the answer paragraph.
You can find the source code in [SentimentAnalysis.java](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/java/ai/djl/examples/inference/SentimentAnalysis.java).

Example:

Input text for analysis

```text
I like DJL. DJL is the best DL framework!
```

Result:

```text
[
	class: "Positive", probability: 0.99864
	class: "Negative", probability: 0.00135
]
```

## Setup guide

Follow [setup](../../docs/development/setup.md) to configure your development environment.

### Run Inference

```sh
cd examples
./gradlew run -Dmain=ai.djl.examples.inference.SentimentAnalysis
```

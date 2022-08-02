# Speech Recognition Example

NOTE: At the time of development, this does not run natively on arm Macs.

In this example, you learn how to use Speech Recognition using PyTorch.
You can provide the model with a wav input file. 
You can find the source code in [SpeechRecognition.java](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/java/ai/djl/examples/inference/SpeechRecognition.java).

Example:

Input audio

Result:

```text
[
    Result: 
]
```

## Setup guide

Follow [setup](../../docs/development/setup.md) to configure your development environment.

### Run Inference

```
cd examples
./gradlew run -Dmain=ai.djl.examples.inference.SpeechRecognition
```
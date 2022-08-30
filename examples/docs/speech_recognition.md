# Speech Recognition Example

In this example, you learn how to use Speech Recognition using PyTorch.
You can provide the model with a wav input file. 
You can find the source code in [SpeechRecognition.java](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/java/ai/djl/examples/inference/SpeechRecognition.java).

Example:

Input audio: https://resources.djl.ai/audios/speech.wav

Result:

```text
THE NEAREST SAID THE DISTRICT DOCTOR IS A GOOD ITALIAN ABBE WHO LIVES NEXT DOOR TO YOU SHALL I CALL ON HIM AS I PASS 
```

## Setup guide

Follow [setup](../../docs/development/setup.md) to configure your development environment.

### Run Inference

```
cd examples
./gradlew run -Dmain=ai.djl.examples.inference.SpeechRecognition

[INFO ] - Result: THE NEAREST SAID THE DISTRICT DOCTOR IS A GOOD ITALIAN ABBE WHO LIVES NEXT DOOR TO YOU SHALL I CALL ON HIM AS I PASS 
```

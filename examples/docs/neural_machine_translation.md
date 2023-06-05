# Neural Machine Translation Example

NOTE: At the time of development, this does not run natively on arm Macs.

In this example, you learn how to use the Neural Machine Translation using PyTorch.
You can provide the model with French text. The model is then able to determine the english meaning from your French 
text input. The model first encodes the French input text, mapping to French words. Then it runs through a different
model to decode the French words to English words.
You can find the source code in [NeuralMachineTranslation.java](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/java/ai/djl/examples/inference/NeuralMachineTranslation.java).

Example:

Input text for analysis

```text
trop tard
```

Result:

```text
[
    French: trop tard
    English: i m too late
]
```

## Setup guide

Follow [setup](../../docs/development/setup.md) to configure your development environment.

### Run Inference

```sh
cd examples
./gradlew run -Dmain=ai.djl.examples.inference.NeuralMachineTranslation
```


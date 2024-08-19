# OpenAI Whipser model in DJL

[Whisper](https://github.com/openai/whisper) is an open source model released by OpenAI.
It can do speech recognition and also machine translation within a single model. In this tutorial,
we just convert the English portion of the model into Java.

Our input is an audio file:

```
https://resources.djl.ai/audios/jfk.flac
```

## Run the example

```sh
cd examples
./gradlew run -Dmain=ai.djl.examples.inference.whisper.SpeechToTextGeneration
```

Output:

```
<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|> And so my fellow Americans , ask not what your country can do for you , ask what you can do for your country . <|endoftext|>
```

## Trace the model

You can use the following script to get the traced model and also vocabulary files, this script works
with `transformers==4.38.0`

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torch
import numpy as np

processor = WhisperProcessor.from_pretrained("openai/whisper-base")
processor.tokenizer.save_pretrained("whisper-tokenizer")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base", torchscript=True, attn_implementation="eager")
model.generation_config.language = "en"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

test = []
for ele in ds:
    test.append(ele["audio"]["array"])

input_features = processor(np.concatenate(test), return_tensors="pt").input_features
generated_ids = model.generate(inputs=input_features)
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Original: {transcription}")

# Start tracing
traced_model = torch.jit.trace_module(model, {"generate": [input_features]})
generated_ids = traced_model.generate(input_features)
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Traced: {transcription}")

torch.jit.save(traced_model, "whisper_en.pt")
```

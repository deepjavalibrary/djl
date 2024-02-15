# Resource Caches

DJL uses cache directories to store downloaded models and Engine specific native files.
By default, cache directories are located at current user's home directory:

- `.djl.ai` is the default cache directory stores downloaded models and datasets
- `.djl.ai/mxnet` is the default cache directory stores Apache MXNet engine native libraries
- `.djl.ai/pytorch` is the default cache directory stores Pytorch engine native libraries
- `.djl.ai/tensorflow` is the default cache directory stores TensorFlow engine native libraries
- `.djl.ai/fasttext` is the default cache directory stores fastText native libraries
- `.djl.ai/sentencepiece` is the default cache directory stores Sentencepiece native libraries

*In the current version, DJL will not clean obsolete cache automatically in the current version.*
User can clean up unused model or native engine manually.

Users may need change cache directory location in some cases. For example, sometimes users may
have limited access to this directory (Read Only) or user's home directory doesn't have enough disk space. 
To avoid downloading failures in these situations, users can specify a custom location to use instead:

- `DJL_CACHE_DIR` is a system property or environment variable you can set to change the global cache location.
Changing this variable will change the location for both model and engine native files.
- `ENGINE_CACHE_DIR` is a system property or environment variable you can set to change the Engine cache location.
For this option, the model directory won't change unless you also change the `DJL_CACHE_DIR`.

## Other cache folders

### ONNXRuntime

ONNXRuntime will extract native libraries into system default temporary-file directory. 

### Huggingface tokenizer

If the `HF_HOME` or `HF_HUB_CACHE` environment variable is set, Huggingface tokenizer will store cache files in it.
It is the responsibility of the user to make sure this path is correct. Otherwise, we try to use
the default cache directory as defined for each OS:

- macOS: `/Users/{user}/.cache/huggingface/hub`
- linux: `/home/{user}/.cache/huggingface/hub`
- windows: `C:\Users\{user}\.cache\huggingface\hub`

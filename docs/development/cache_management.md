# Cache Management

DJL uses cache directories to store downloaded models and Engine specific native files.
By default, cache directories are located at current user's home directory:

- `.djl.ai` is the default cache directory stores downloaded models and datasets
- `.mxnet` is the default cache directory stores Apache MXNet engine native libraries
- `.pytorch` is the default cache directory stores Pytorch engine native libraries
- `.tensorflow` is the default cache directory stores TensorFlow engine native libraries

*In the current version, DJL will not clean obsolete cache automatically in the current version.*
User can clean up unused model or native engine manually.

Users may need change cache directory location in some cases. For example, sometimes users may
have limited access to this directory (Read Only) or user's home directory doesn't have enough disk space. 
To avoid downloading failures in these situations, users can specify a custom location to use instead:

- `DJL_CACHE_DIR` is a system property or environment variable you can set to change the global cache location.
Changing this variable will change the location for both model and engine native files.
- `ENGINE_CACHE_DIR` is a system property or environment variable you can set to change the Engine cache location.
For this option, the model directory won't change unless you also change the `DJL_CACHE_DIR`.


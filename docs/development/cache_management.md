# DJL Cache Management

By default, DJL will use a cache directory to store downloaded models 
and Engine specific native files. By default, models are in `.djl.ai`, engines are cached in `.<engine-name>/` (e.g `.pytorch/`) folder
However, sometimes you may have limited access to this directory (Read Only) which causes failures when downloading models and loading engines.
In this situation, you can specify a custom location to use instead.

- `DJL_CACHE_DIR` is a system property or environment variable you can set to change the global cache location.
Changing this variable will change the location for both model and engine native files.
- `ENGINE_CACHE_DIR` is a system property or environment variable you can set to change the Engine cache location.
For this option, the model directory won't change unless you also change the `DJL_CACHE_DIR`.

DJL will not clean obsolete cache automatically in the current version.
User can clean up unused model or native engine manually.

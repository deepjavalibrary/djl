# DeepJavaLibrary - MXNet engine implementation

## Overview

This module contains MXNet implementation of djl.ai EngineProvider.

It's not recommended for developer to use classes in this module directly. It will make your code
tight with MXNet. However we do provide a way that allows you call Engine specific API. See
[NDManager#invoke()](https://djl-ai.s3.amazonaws.com/java-api/0.2.0/ai/djl/ndarray/NDManager.html#invoke-java.lang.String-ai.djl.ndarray.NDList-ai.djl.ndarray.NDList-ai.djl.util.PairList-)
for detail.

# DJL - Tensorflow engine implementation

## Overview

This module contains the Tensorflow implementation of the Deep Java Library (DJL) EngineProvider.

We don't recommend that developers use classes in this module directly. Use of these classes will couple your code with Tensorflow and make switching between frameworks difficult. Even so, developers are not restricted from using engine-specific features. For more information, see [NDManager#invoke()](https://javadoc.djl.ai/api/0.2.0/ai/djl/ndarray/NDManager.html#invoke-java.lang.String-ai.djl.ndarray.NDList-ai.djl.ndarray.NDList-ai.djl.util.PairList-).

**Right now, the tensorflow API is here as a proof of concept. While it can help provide a starting point towards a full Tensorflow implementation, it should not be used in it's current state.**

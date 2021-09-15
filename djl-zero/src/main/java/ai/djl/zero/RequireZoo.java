/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.zero;

import ai.djl.engine.Engine;
import ai.djl.repository.zoo.ModelZoo;

/**
 * A set of utilities for requiring a {@link ModelZoo}.
 *
 * <p>Throws an exception if the {@link ModelZoo} is not available.
 */
public final class RequireZoo {

    private RequireZoo() {}

    /** Requires {@code ai.djl.basicmodelzoo.BasicModelZoo}. */
    public static void basic() {
        if (!ModelZoo.hasModelZoo("ai.djl.zoo")) {
            throw new IllegalStateException(
                    "The basic model zoo is required, but not found."
                            + "Please install it by following http://docs.djl.ai/model-zoo/index.html#installation");
        }
    }

    /** Requires {@code ai.djl.mxnet.zoo.MxModelZoo}. */
    public static void mxnet() {
        if (!ModelZoo.hasModelZoo("ai.djl.mxnet")) {
            throw new IllegalStateException(
                    "The MXNet model zoo is required, but not found."
                            + "Please install it by following http://docs.djl.ai/engines/mxnet/mxnet-model-zoo/index.html#installation");
        }
        if (!Engine.hasEngine("MXNet")) {
            throw new IllegalStateException(
                    "The MXNet engine is required, but not found."
                            + "Please install it by following http://docs.djl.ai/engines/mxnet/mxnet-engine/index.html#installation");
        }
    }

    /** Requires {@code ai.djl.pytorch.zoo.PtModelZoo}. */
    public static void pytorch() {
        if (!ModelZoo.hasModelZoo("ai.djl.pytorch")) {
            throw new IllegalStateException(
                    "The PyTorch model zoo is required, but not found."
                            + "Please install it by following http://docs.djl.ai/pytorch/pytorch-model-zoo/index.html#installation");
        }
        if (!Engine.hasEngine("PyTorch")) {
            throw new IllegalStateException(
                    "The PyTorch engine is required, but not found."
                            + "Please install it by following http://docs.djl.ai/pytorch/pytorch-engine/index.html#installation");
        }
    }

    /** Requires {@code ai.djl.tensorflow.zoo.TfModelZoo}. */
    public static void tensorflow() {
        if (!ModelZoo.hasModelZoo("ai.djl.tensorflow")) {
            throw new IllegalStateException(
                    "The TensorFlow model zoo is required, but not found."
                            + "Please install it by following http://docs.djl.ai/engines/tensorflow/tensorflow-model-zoo/index.html#installation");
        }
        if (!Engine.hasEngine("TensorFlow")) {
            throw new IllegalStateException(
                    "The TensorFlow engine is required, but not found."
                            + "Please install it by following http://docs.djl.ai/engines/tensorflow/tensorflow-engine/index.html#installation");
        }
    }
}

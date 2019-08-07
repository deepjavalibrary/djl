/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package software.amazon.ai.training;

import software.amazon.ai.Block;
import software.amazon.ai.Parameter;
import software.amazon.ai.engine.Engine;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.util.PairList;

/**
 * A collection of utilities to collect and retrieve gradients. Use {@link Gradient#newCollector()}
 * to start.
 */
public interface Gradient {

    static Collector newCollector() {
        return Engine.getInstance().newGradientCollector();
    }

    /**
     * The gradient collector should be opened with a try-with-resources. First call collectFor on
     * whatever data you want to collect gradients for, then execute the operations. Call
     * collectProgress to execute backwards multiple times and collect to execute backwards for the
     * final time.
     */
    interface Collector extends AutoCloseable {

        OptimizerKey collectFor(Optimizer optimizer);

        BlockKey collectFor(Block block);

        ParameterKey collectFor(Parameter parameter);

        NDArrayKey collectFor(NDArray array);

        void collectProgress(NDArray target);

        Dict collect(NDArray target);

        /** {@inheritDoc} */
        @Override
        void close();
    }

    /**
     * A collection of Gradients that can be retrieved from with the key created when calling the
     * {@code Gradient.Collector.collectFor} method.
     */
    interface Dict {

        OptimizerGrad get(OptimizerKey key);

        BlockGrad get(BlockKey key);

        ParameterGrad get(ParameterKey key);

        NDArrayGrad get(NDArrayKey key);
    }

    /** An abstract key to be used to retrieve gradients from a {@link Gradient.Dict}. */
    interface Key {}

    /** A Key to retrieve gradients for an optimizer from a {@link Gradient.Dict}. */
    interface OptimizerKey extends Key {
        Optimizer getOptimizer();
    }

    /** The gradients for an optimizer. */
    interface OptimizerGrad extends Gradient {
        Optimizer getOptimizer();

        PairList<String, NDArray> get();
    }

    /** A Key to retrieve gradients for a block from a {@link Gradient.Dict}. */
    interface BlockKey extends Key {
        Block getBlock();
    }

    /** The gradients for a block. */
    interface BlockGrad extends Gradient {
        Block getBlock();

        PairList<String, NDArray> get();
    }

    /** A Key to retrieve gradients for a parameter from a {@link Gradient.Dict}. */
    interface ParameterKey extends Key {
        Parameter getParameter();
    }

    /** The gradients for a parameter. */
    interface ParameterGrad extends Gradient {
        Parameter getParameter();

        NDArray get();
    }

    /** A Key to retrieve gradients for an NDArray from a {@link Gradient.Dict}. */
    interface NDArrayKey extends Key {
        NDArray getArray();
    }

    /** The gradients for an NDArray. */
    interface NDArrayGrad extends Gradient {
        NDArray getArray();

        NDArray get();
    }
}

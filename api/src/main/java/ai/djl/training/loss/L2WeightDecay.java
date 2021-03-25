/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.training.loss;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;

/**
 * {@code L2WeightDecay} calculates L2 penalty of a set of parameters. Used for regularization.
 *
 * <p>L2 loss is defined by \(L2 = \lambda \sum_i {W_i}^2\).
 */
public class L2WeightDecay extends Loss {

    private float lambda;
    private NDList parameters;

    /**
     * Calculates L2 weight decay for regularization.
     *
     * @param parameters holds the model weights that will be penalized
     */
    public L2WeightDecay(NDList parameters) {
        this("L2WeightDecay", parameters);
    }

    /**
     * Calculates L2 weight decay for regularization.
     *
     * @param name the name of the penalty
     * @param parameters holds the model weights that will be penalized
     */
    public L2WeightDecay(String name, NDList parameters) {
        this(name, parameters, 1);
    }

    /**
     * Calculates L2 weight decay for regularization.
     *
     * @param name the name of the penalty
     * @param parameters holds the model weights that will be penalized
     * @param lambda the weight to apply to the penalty value, default 1
     */
    public L2WeightDecay(String name, NDList parameters, float lambda) {
        super(name);
        this.lambda = lambda;
        this.parameters = parameters;
    }

    private NDArray l2(NDArray w) {
        return ((w.square()).sum());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray evaluate(NDList label, NDList prediction) {

        NDManager manager = parameters.getManager();
        NDArray sum = manager.create(0.0f);
        for (NDArray wi : parameters) {
            sum.addi(l2(wi));
        }
        return sum.muli(lambda);
    }
}

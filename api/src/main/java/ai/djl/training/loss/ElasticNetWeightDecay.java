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
 * {@code ElasticWeightDecay} calculates L1+L2 penalty of a set of parameters. Used for
 * regularization.
 *
 * <p>L loss is defined as \(L = \lambda_1 \sum_i \vert W_i\vert + \lambda_2 \sum_i {W_i}^2\).
 */
public class ElasticNetWeightDecay extends Loss {

    private float lambda1;
    private float lambda2;
    private NDList parameters;

    /**
     * Calculates Elastic Net weight decay for regularization.
     *
     * @param parameters holds the model weights that will be penalized
     */
    public ElasticNetWeightDecay(NDList parameters) {
        this("ElasticNetWeightDecay", parameters);
    }

    /**
     * Calculates Elastic Net weight decay for regularization.
     *
     * @param name the name of the penalty
     * @param parameters holds the model weights that will be penalized
     */
    public ElasticNetWeightDecay(String name, NDList parameters) {
        this(name, parameters, 1);
    }

    /**
     * Calculates Elastic Net weight decay for regularization.
     *
     * @param name the name of the penalty
     * @param parameters holds the model weights that will be penalized
     * @param lambda the weight to apply to the penalty value, default 1 (both L1 and L2)
     */
    public ElasticNetWeightDecay(String name, NDList parameters, float lambda) {
        super(name);
        this.lambda1 = lambda;
        this.lambda2 = lambda;
        this.parameters = parameters;
    }

    /**
     * Calculates Elastic Net weight decay for regularization.
     *
     * @param name the name of the penalty
     * @param parameters holds the model weights that will be penalized
     * @param lambda1 the weight to apply to the L1 penalty value, default 1
     * @param lambda2 the weight to apply to the L2 penalty value, default 1
     */
    public ElasticNetWeightDecay(String name, NDList parameters, float lambda1, float lambda2) {
        super(name);
        this.lambda1 = lambda1;
        this.lambda2 = lambda2;
        this.parameters = parameters;
    }

    private NDArray l1(NDArray w) {
        return ((w.abs()).sum());
    }

    private NDArray l2(NDArray w) {
        return ((w.square()).sum());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray evaluate(NDList label, NDList prediction) {

        NDManager manager = parameters.getManager();
        NDArray sum1 = manager.create(0.0f);
        NDArray sum2 = manager.create(0.0f);
        for (NDArray wi : parameters) {
            sum1.addi(l1(wi));
            sum2.addi(l2(wi));
        }
        return sum1.muli(lambda1).addi(sum2.muli(lambda2));
    }
}

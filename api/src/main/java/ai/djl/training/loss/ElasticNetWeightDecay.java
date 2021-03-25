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
 * {@code ElasticWeightDecay} calculates L1+L2 penalty of a set of parameters. Used for regularization.
 * 
 * <p>L loss is defined by \(L = \lambda \sum_i \vert W_i\vert + \lambda \sum_i \vert {W_i}^2\vert\).
 */
public class ElasticNetWeightDecay extends Loss {

    private float lambda1;
    private float lambda2;
    private NDList parameters;

    /** Calculates L2 weight decay for regularization. */
    public ElasticNetWeightDecay(NDList parameters) {
        this("ElasticNetWeightDecay", parameters);
    }

    /**
     * Calculates L2 weight decay for regularization.
     *
     * @param name the name of the penalty
     */
    public ElasticNetWeightDecay(String name, NDList parameters) {
        this(name, parameters, 1);
    }

    /**
     * Calculates L2 weight decay for regularization.
     *
     * @param name the name of the penalty
     * @param weight the weight to apply on penalty value, default 1
     */
    public ElasticNetWeightDecay(String name, NDList parameters, float lambda) {
        super(name);
        this.lambda1 = lambda;
        this.lambda2 = lambda;
        this.parameters = parameters;
    }

    /**
     * Calculates L2 weight decay for regularization.
     *
     * @param name the name of the penalty
     * @param weight the weight to apply on penalty value, default 1
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
        for(NDArray wi : parameters){
            sum1.addi( l1(wi) );
            sum2.addi( l2(wi) );
        }
        return sum1.muli(lambda1).addi( sum2.muli(lambda2));
    }
}

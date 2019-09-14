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
package org.apache.mxnet.nn;

import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.nn.Block;
import software.amazon.ai.nn.LambdaBlock;
import software.amazon.ai.nn.core.Prelu;
import software.amazon.ai.training.Activation;

public class MxActivation implements Activation {

    private NDManager manager;

    public MxActivation(NDManager manager) {
        this.manager = manager;
    }

    @Override
    public Block reluBlock() {
        return new LambdaBlock(manager, Activation::relu);
    }

    @Override
    public Block sigmoidBlock() {
        return new LambdaBlock(manager, Activation::sigmoid);
    }

    @Override
    public Block tanhBlock() {
        return new LambdaBlock(manager, Activation::tanh);
    }

    @Override
    public Block softreluBlock() {
        return new LambdaBlock(manager, Activation::softrelu);
    }

    @Override
    public Block leakyReluBlock(float alpha) {
        return new LambdaBlock(manager, arrays -> Activation.leakyRelu(arrays, alpha));
    }

    @Override
    public Block eluBlock(float alpha) {
        return new LambdaBlock(manager, arrays -> Activation.elu(arrays, alpha));
    }

    @Override
    public Block seluBlock() {
        return new LambdaBlock(manager, Activation::selu);
    }

    @Override
    public Block geluBlock() {
        return new LambdaBlock(manager, Activation::gelu);
    }

    @Override
    public Block swishBlock(float beta) {
        return new LambdaBlock(manager, arrays -> Activation.swish(arrays, beta));
    }

    @Override
    public Block preluBlock() {
        return new Prelu(manager);
    }
}

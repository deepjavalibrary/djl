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

import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.nn.Block;

public interface Activation {

    static NDArray relu(NDArray array) {
        return array.getNDArrayInternal().relu();
    }

    static NDList relu(NDList arrays) {
        return new NDList(arrays.get(0).getNDArrayInternal().relu());
    }

    static NDArray sigmoid(NDArray array) {
        return array.getNDArrayInternal().sigmoid();
    }

    static NDList sigmoid(NDList arrays) {
        return new NDList(arrays.get(0).getNDArrayInternal().sigmoid());
    }

    static NDArray tanh(NDArray array) {
        return array.getNDArrayInternal().tanh();
    }

    static NDList tanh(NDList arrays) {
        return new NDList(arrays.get(0).getNDArrayInternal().tanh());
    }

    static NDArray softrelu(NDArray array) {
        return array.getNDArrayInternal().softrelu();
    }

    static NDList softrelu(NDList arrays) {
        return new NDList(arrays.get(0).getNDArrayInternal().softrelu());
    }

    static NDArray leakyRelu(NDArray array, float alpha) {
        return array.getNDArrayInternal().leakyRelu(alpha);
    }

    static NDList leakyRelu(NDList arrays, float alpha) {
        return new NDList(arrays.get(0).getNDArrayInternal().leakyRelu(alpha));
    }

    static NDArray elu(NDArray array, float alpha) {
        return array.getNDArrayInternal().elu(alpha);
    }

    static NDList elu(NDList arrays, float alpha) {
        return new NDList(arrays.get(0).getNDArrayInternal().elu(alpha));
    }

    static NDArray selu(NDArray array) {
        return array.getNDArrayInternal().selu();
    }

    static NDList selu(NDList arrays) {
        return new NDList(arrays.get(0).getNDArrayInternal().selu());
    }

    static NDArray gelu(NDArray array) {
        return array.getNDArrayInternal().gelu();
    }

    static NDList gelu(NDList arrays) {
        return new NDList(arrays.get(0).getNDArrayInternal().gelu());
    }

    static NDArray swish(NDArray array, float beta) {
        return array.getNDArrayInternal().swish(beta);
    }

    static NDList swish(NDList arrays, float beta) {
        return new NDList(arrays.get(0).getNDArrayInternal().swish(beta));
    }

    Block reluBlock();

    Block sigmoidBlock();

    Block tanhBlock();

    Block softreluBlock();

    Block leakyReluBlock(float alpha);

    Block eluBlock(float alpha);

    Block seluBlock();

    Block geluBlock();

    Block swishBlock(float beta);

    Block preluBlock();
}

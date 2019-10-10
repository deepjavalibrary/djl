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
package ai.djl.training;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.core.Prelu;

public interface Activation {

    Block IDENTITY_BLOCK = new LambdaBlock(x -> x);

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

    static Block reluBlock() {
        return new LambdaBlock(Activation::relu);
    }

    static Block sigmoidBlock() {
        return new LambdaBlock(Activation::sigmoid);
    }

    static Block tanhBlock() {
        return new LambdaBlock(Activation::tanh);
    }

    static Block softreluBlock() {
        return new LambdaBlock(Activation::softrelu);
    }

    static Block leakyReluBlock(float alpha) {
        return new LambdaBlock(arrays -> Activation.leakyRelu(arrays, alpha));
    }

    static Block eluBlock(float alpha) {
        return new LambdaBlock(arrays -> Activation.elu(arrays, alpha));
    }

    static Block seluBlock() {
        return new LambdaBlock(Activation::selu);
    }

    static Block geluBlock() {
        return new LambdaBlock(Activation::gelu);
    }

    static Block swishBlock(float beta) {
        return new LambdaBlock(arrays -> Activation.swish(arrays, beta));
    }

    static Block preluBlock() {
        return new Prelu();
    }
}

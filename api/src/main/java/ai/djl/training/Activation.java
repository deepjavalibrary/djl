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

public final class Activation {

    public static final Block IDENTITY_BLOCK = new LambdaBlock(x -> x);

    private Activation() {}

    public static NDArray relu(NDArray array) {
        return array.getNDArrayInternal().relu();
    }

    public static NDList relu(NDList arrays) {
        return new NDList(arrays.get(0).getNDArrayInternal().relu());
    }

    public static NDArray sigmoid(NDArray array) {
        return array.getNDArrayInternal().sigmoid();
    }

    public static NDList sigmoid(NDList arrays) {
        return new NDList(arrays.get(0).getNDArrayInternal().sigmoid());
    }

    public static NDArray tanh(NDArray array) {
        return array.getNDArrayInternal().tanh();
    }

    public static NDList tanh(NDList arrays) {
        return new NDList(arrays.get(0).getNDArrayInternal().tanh());
    }

    public static NDArray softrelu(NDArray array) {
        return array.getNDArrayInternal().softrelu();
    }

    public static NDList softrelu(NDList arrays) {
        return new NDList(arrays.get(0).getNDArrayInternal().softrelu());
    }

    public static NDArray leakyRelu(NDArray array, float alpha) {
        return array.getNDArrayInternal().leakyRelu(alpha);
    }

    public static NDList leakyRelu(NDList arrays, float alpha) {
        return new NDList(arrays.get(0).getNDArrayInternal().leakyRelu(alpha));
    }

    public static NDArray elu(NDArray array, float alpha) {
        return array.getNDArrayInternal().elu(alpha);
    }

    public static NDList elu(NDList arrays, float alpha) {
        return new NDList(arrays.get(0).getNDArrayInternal().elu(alpha));
    }

    public static NDArray selu(NDArray array) {
        return array.getNDArrayInternal().selu();
    }

    public static NDList selu(NDList arrays) {
        return new NDList(arrays.get(0).getNDArrayInternal().selu());
    }

    public static NDArray gelu(NDArray array) {
        return array.getNDArrayInternal().gelu();
    }

    public static NDList gelu(NDList arrays) {
        return new NDList(arrays.get(0).getNDArrayInternal().gelu());
    }

    public static NDArray swish(NDArray array, float beta) {
        return array.getNDArrayInternal().swish(beta);
    }

    public static NDList swish(NDList arrays, float beta) {
        return new NDList(arrays.get(0).getNDArrayInternal().swish(beta));
    }

    public static Block reluBlock() {
        return new LambdaBlock(Activation::relu);
    }

    public static Block sigmoidBlock() {
        return new LambdaBlock(Activation::sigmoid);
    }

    public static Block tanhBlock() {
        return new LambdaBlock(Activation::tanh);
    }

    public static Block softreluBlock() {
        return new LambdaBlock(Activation::softrelu);
    }

    public static Block leakyReluBlock(float alpha) {
        return new LambdaBlock(arrays -> Activation.leakyRelu(arrays, alpha));
    }

    public static Block eluBlock(float alpha) {
        return new LambdaBlock(arrays -> Activation.elu(arrays, alpha));
    }

    public static Block seluBlock() {
        return new LambdaBlock(Activation::selu);
    }

    public static Block geluBlock() {
        return new LambdaBlock(Activation::gelu);
    }

    public static Block swishBlock(float beta) {
        return new LambdaBlock(arrays -> Activation.swish(arrays, beta));
    }

    public static Block preluBlock() {
        return new Prelu();
    }
}

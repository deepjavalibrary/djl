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

import java.util.function.Function;
import software.amazon.ai.engine.Engine;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.nn.Block;
import software.amazon.ai.nn.LambdaBlock;

public interface Activation extends Block {

    static NDArray relu(NDArray array) {
        return array.getNDArrayInternal().relu();
    }

    static NDList relu(NDList arrays) {
        return new NDList(arrays.get(0).getNDArrayInternal().relu());
    }

    static Activation reluBlock() {
        return new ActivationLambdaBlock(Activation::relu);
    }

    static NDArray sigmoid(NDArray array) {
        return array.getNDArrayInternal().sigmoid();
    }

    static NDList sigmoid(NDList arrays) {
        return new NDList(arrays.get(0).getNDArrayInternal().sigmoid());
    }

    static Activation sigmoidBlock() {
        return new ActivationLambdaBlock(Activation::sigmoid);
    }

    static NDArray tanh(NDArray array) {
        return array.getNDArrayInternal().tanh();
    }

    static NDList tanh(NDList arrays) {
        return new NDList(arrays.get(0).getNDArrayInternal().tanh());
    }

    static Activation tanhBlock() {
        return new ActivationLambdaBlock(Activation::tanh);
    }

    static NDArray softrelu(NDArray array) {
        return array.getNDArrayInternal().softrelu();
    }

    static NDList softrelu(NDList arrays) {
        return new NDList(arrays.get(0).getNDArrayInternal().softrelu());
    }

    static Activation softreluBlock() {
        return new ActivationLambdaBlock(Activation::softrelu);
    }

    static NDArray leakyRelu(NDArray array, float alpha) {
        return array.getNDArrayInternal().leakyRelu(alpha);
    }

    static NDList leakyRelu(NDList arrays, float alpha) {
        return new NDList(arrays.get(0).getNDArrayInternal().leakyRelu(alpha));
    }

    static Activation leakyReluBlock(float alpha) {
        return new ActivationLambdaBlock(arrays -> leakyRelu(arrays, alpha));
    }

    static NDArray elu(NDArray array, float alpha) {
        return array.getNDArrayInternal().elu(alpha);
    }

    static NDList elu(NDList arrays, float alpha) {
        return new NDList(arrays.get(0).getNDArrayInternal().elu(alpha));
    }

    static Activation eluBlock(float alpha) {
        return new ActivationLambdaBlock(arrays -> elu(arrays, alpha));
    }

    static NDArray selu(NDArray array) {
        return array.getNDArrayInternal().selu();
    }

    static NDList selu(NDList arrays) {
        return new NDList(arrays.get(0).getNDArrayInternal().selu());
    }

    static Activation seluBlock() {
        return new ActivationLambdaBlock(Activation::selu);
    }

    static NDArray gelu(NDArray array) {
        return array.getNDArrayInternal().gelu();
    }

    static NDList gelu(NDList arrays) {
        return new NDList(arrays.get(0).getNDArrayInternal().gelu());
    }

    static Activation geluBlock() {
        return new ActivationLambdaBlock(Activation::gelu);
    }

    static NDArray swish(NDArray array, float beta) {
        return array.getNDArrayInternal().swish(beta);
    }

    static NDList swish(NDList arrays, float beta) {
        return new NDList(arrays.get(0).getNDArrayInternal().swish(beta));
    }

    static Activation swishBlock(float beta) {
        return new ActivationLambdaBlock(arrays -> swish(arrays, beta));
    }

    static Activation preluBlock() {
        return Engine.getInstance().getNNIndex().prelu();
    }

    NDArray forward(NDArray data);

    final class ActivationLambdaBlock extends LambdaBlock implements Activation {

        public ActivationLambdaBlock(Function<NDList, NDList> lambda) {
            super(lambda);
        }

        @Override
        public NDArray forward(NDArray data) {
            return forward(new NDList(data)).get(0);
        }
    }
}

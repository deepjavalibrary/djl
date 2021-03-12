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
package ai.djl.nn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.nn.core.Prelu;

/**
 * Utility class that provides activation functions and blocks.
 *
 * <p>Many networks make use of the {@link ai.djl.nn.core.Linear} block and other similar linear
 * transformations. However, any number of linear transformations that are composed will only result
 * in a different linear transformation (\($f(x) = W_2(W_1x) = (W_2W_1)x = W_{combined}x\)). In
 * order to represent non-linear data, non-linear functions called activation functions are
 * interspersed between the linear transformations. This allows the network to represent non-linear
 * functions of increasing complexity.
 *
 * <p>See <a href="https://en.wikipedia.org/wiki/Activation_function">wikipedia</a> for more
 * details.
 */
public final class Activation {

    private Activation() {}

    /**
     * Applies ReLU activation on the input {@link NDArray}.
     *
     * <p>ReLU is defined by: \( y = max(0, x) \)
     *
     * @param array the input {@link NDArray}
     * @return the {@link NDArray} after applying ReLU activation
     */
    public static NDArray relu(NDArray array) {
        return array.getNDArrayInternal().relu();
    }

    /**
     * Applies ReLU activation on the input singleton {@link NDList}.
     *
     * <p>ReLU is defined by: \( y = max(0, x) \)
     *
     * @param arrays the input singleton {@link NDList}
     * @return the singleton {@link NDList} after applying ReLU activation
     */
    public static NDList relu(NDList arrays) {
        return new NDList(arrays.singletonOrThrow().getNDArrayInternal().relu());
    }

    /**
     * Applies Sigmoid activation on the input {@link NDArray}.
     *
     * <p>Sigmoid is defined by: \( y = 1 / (1 + e^{-x}) \)
     *
     * @param array the input {@link NDArray}
     * @return the {@link NDArray} after applying Sigmoid activation
     */
    public static NDArray sigmoid(NDArray array) {
        return array.getNDArrayInternal().sigmoid();
    }

    /**
     * Applies Sigmoid activation on the input singleton {@link NDList}.
     *
     * <p>Sigmoid is defined by: \( y = 1 / (1 + e^{-x}) \)
     *
     * @param arrays the input singleton {@link NDList}
     * @return the singleton {@link NDList} after applying Sigmoid activation
     */
    public static NDList sigmoid(NDList arrays) {
        return new NDList(arrays.singletonOrThrow().getNDArrayInternal().sigmoid());
    }

    /**
     * Applies Tanh activation on the input {@link NDArray}.
     *
     * <p>Tanh is defined by: \( y = (e^x - e^{-x}) / (e^x + e^{-x}) \)
     *
     * @param array the input {@link NDArray}
     * @return the {@link NDArray} after applying Tanh activation
     */
    public static NDArray tanh(NDArray array) {
        return array.getNDArrayInternal().tanh();
    }

    /**
     * Applies Tanh activation on the input singleton {@link NDList}.
     *
     * <p>Tanh is defined by: \( y = (e^x - e^{-x}) / (e^x + e^{-x}) \)
     *
     * @param arrays the input singleton {@link NDList}
     * @return the singleton {@link NDList} after applying tanh activation
     */
    public static NDList tanh(NDList arrays) {
        return new NDList(arrays.singletonOrThrow().getNDArrayInternal().tanh());
    }

    /**
     * Applies softPlus activation on the input {@link NDArray}.
     *
     * <p>softPlus is defined by: \( y = log(1 + e^x) \)
     *
     * @param array the input {@link NDArray}
     * @return the {@link NDArray} after applying soft ReLU activation
     */
    public static NDArray softPlus(NDArray array) {
        return array.getNDArrayInternal().softPlus();
    }

    /**
     * Applies softPlus activation on the input singleton {@link NDList}.
     *
     * <p>softPlus is defined by: \( y = log(1 + e^x) \)
     *
     * @param arrays the input singleton {@link NDList}
     * @return the singleton {@link NDList} after applying soft ReLU activation
     */
    public static NDList softPlus(NDList arrays) {
        return new NDList(arrays.singletonOrThrow().getNDArrayInternal().softPlus());
    }

    /**
     * Applies softSign activation on the input {@link NDArray}.
     *
     * <p>softPlus is defined by: \( y = x / 1 + |x| \)
     *
     * @param array the input {@link NDArray}
     * @return the {@link NDArray} after applying soft ReLU activation
     */
    public static NDArray softSign(NDArray array) {
        return array.getNDArrayInternal().softSign();
    }

    /**
     * Applies softPlus activation on the input singleton {@link NDList}.
     *
     * <p>softPlus is defined by: \( y = x / 1 + |x| \)
     *
     * @param arrays the input singleton {@link NDList}
     * @return the singleton {@link NDList} after applying soft ReLU activation
     */
    public static NDList softSign(NDList arrays) {
        return new NDList(arrays.singletonOrThrow().getNDArrayInternal().softSign());
    }

    /**
     * Applies Leaky ReLU activation on the input {@link NDArray}.
     *
     * <p>Leaky ReLU is defined by: \( y = x \gt 0 ? x : alpha * x \)
     *
     * @param array the input {@link NDArray}
     * @param alpha the slope for the activation
     * @return the {@link NDArray} after applying Leaky ReLU activation
     */
    public static NDArray leakyRelu(NDArray array, float alpha) {
        return array.getNDArrayInternal().leakyRelu(alpha);
    }

    /**
     * Applies Leaky ReLU activation on the input singleton {@link NDList}.
     *
     * <p>Leaky ReLU is defined by: \( y = x \gt 0 ? x : alpha * x \)
     *
     * @param arrays the input singleton {@link NDList}
     * @param alpha the slope for the activation
     * @return the singleton {@link NDList} after applying Leaky ReLU activation
     */
    public static NDList leakyRelu(NDList arrays, float alpha) {
        return new NDList(arrays.singletonOrThrow().getNDArrayInternal().leakyRelu(alpha));
    }

    /**
     * Applies ELU activation on the input {@link NDArray}.
     *
     * <p>ELU is defined by: \( y = x \gt 0 ? x : alpha * (e^x - 1) \)
     *
     * @param array the input {@link NDArray}
     * @param alpha the slope for the activation
     * @return the {@link NDArray} after applying ELU activation
     */
    public static NDArray elu(NDArray array, float alpha) {
        return array.getNDArrayInternal().elu(alpha);
    }

    /**
     * Applies ELU(Exponential Linear Unit) activation on the input singleton {@link NDList}.
     *
     * <p>ELU is defined by: \( y = x \gt 0 ? x : alpha * (e^x - 1) \)
     *
     * @param arrays the input singleton {@link NDList}
     * @param alpha the slope for the activation
     * @return the singleton {@link NDList} after applying ELU activation
     */
    public static NDList elu(NDList arrays, float alpha) {
        return new NDList(arrays.singletonOrThrow().getNDArrayInternal().elu(alpha));
    }

    /**
     * Applies Scaled ELU activation on the input {@link NDArray}.
     *
     * <p>Scaled ELU is defined by: \( y = lambda * (x \gt 0 ? x : alpha * (e^x - 1))\) where
     * \(lambda = 1.0507009873554804934193349852946\) and \(alpha =
     * 1.6732632423543772848170429916717\)
     *
     * @param array the input {@link NDArray}
     * @return the {@link NDArray} after applying Scale ELU activation
     */
    public static NDArray selu(NDArray array) {
        return array.getNDArrayInternal().selu();
    }

    /**
     * Applies Scaled ELU activation on the input singleton {@link NDList}.
     *
     * <p>Scaled ELU is defined by: \( y = lambda * (x \gt 0 ? x : alpha * (e^x - 1))\) where
     * \(lambda = 1.0507009873554804934193349852946\) and \(alpha =
     * 1.6732632423543772848170429916717 \)
     *
     * @param arrays the input singleton {@link NDList}
     * @return the singleton {@link NDList} after applying Scaled ELU activation
     */
    public static NDList selu(NDList arrays) {
        return new NDList(arrays.singletonOrThrow().getNDArrayInternal().selu());
    }

    /**
     * Applies GELU(Gaussian Error Linear Unit) activation on the input {@link NDArray}.
     *
     * @param array the input {@link NDArray}
     * @return the {@link NDArray} after applying GELU activation
     */
    public static NDArray gelu(NDArray array) {
        return array.getNDArrayInternal().gelu();
    }

    /**
     * Applies GELU(Gaussian Error Linear Unit) activation on the input singleton {@link NDList}.
     *
     * @param arrays the input singleton {@link NDList}
     * @return the singleton {@link NDList} after applying GELU activation
     */
    public static NDList gelu(NDList arrays) {
        return new NDList(arrays.singletonOrThrow().getNDArrayInternal().gelu());
    }

    /**
     * Applies Swish activation on the input {@link NDArray}.
     *
     * <p>Swish is defined as \(y = x * sigmoid(beta * x)\)
     *
     * @param array the input {@link NDArray}
     * @param beta a hyper-parameter
     * @return the {@link NDArray} after applying Swish activation
     */
    public static NDArray swish(NDArray array, float beta) {
        return array.getNDArrayInternal().swish(beta);
    }

    /**
     * Applies SWish activation on the input singleton {@link NDList}.
     *
     * <p>Swish is defined as \(y = x * sigmoid(beta * x)\)
     *
     * @param arrays the input singleton {@link NDList}
     * @param beta a hyper-parameter
     * @return the singleton {@link NDList} after applying Swish activation
     */
    public static NDList swish(NDList arrays, float beta) {
        return new NDList(arrays.singletonOrThrow().getNDArrayInternal().swish(beta));
    }

    /**
     * Applies Mish activation on the input {@link NDArray}.
     *
     * <p>Mish is defined as \(y = x * tanh(ln(1 + e^x)\) defined by Diganta Misra in his paper
     * Mish: A Self Regularized Non-Monotonic Neural Activation Function
     *
     * @param array the input {@link NDArray}
     * @return the {@link NDArray} after applying Mish activation
     */
    public static NDArray mish(NDArray array) {
        return array.getNDArrayInternal().mish();
    }

    /**
     * Applies Mish activation on the input singleton {@link NDList}.
     *
     * <p>Mish is defined as \(y = x * tanh(ln(1 + e^x)\) defined by Diganta Misra in his paper
     * Mish: A Self Regularized Non-Monotonic Neural Activation Function
     *
     * @param arrays the input singleton {@link NDList}
     * @return the singleton {@link NDList} after applying Mish activation
     */
    public static NDList mish(NDList arrays) {
        return new NDList(arrays.singletonOrThrow().getNDArrayInternal().mish());
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #relu(NDList) ReLU} activation function
     * in its forward function.
     *
     * @return the {@link LambdaBlock} that applies the {@link #relu(NDList) ReLU} activation
     *     function
     */
    public static Block reluBlock() {
        return new LambdaBlock(Activation::relu);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #sigmoid(NDList) Sigmoid} activation
     * function in its forward function.
     *
     * @return the {@link LambdaBlock} that applies the {@link #sigmoid(NDList) Sigmoid} activation
     *     function
     */
    public static Block sigmoidBlock() {
        return new LambdaBlock(Activation::sigmoid);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #tanh(NDList) Tanh} activation function
     * in its forward function.
     *
     * @return the {@link LambdaBlock} that applies the {@link #tanh(NDList) Tanh} activation
     *     function
     */
    public static Block tanhBlock() {
        return new LambdaBlock(Activation::tanh);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #softPlus(NDList)} activation function
     * in its forward function.
     *
     * @return the {@link LambdaBlock} that applies the {@link #softPlus(NDList)} activation
     *     function
     */
    public static Block softPlusBlock() {
        return new LambdaBlock(Activation::softPlus);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #softSign(NDList)} activation function
     * in its forward function.
     *
     * @return the {@link LambdaBlock} that applies the {@link #softSign(NDList)} activation
     *     function
     */
    public static Block softSignBlock() {
        return new LambdaBlock(Activation::softSign);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #leakyRelu(NDList, float) LeakyReLU}
     * activation function in its forward function.
     *
     * @param alpha the slope for the activation
     * @return the {@link LambdaBlock} that applies the {@link #leakyRelu(NDList, float) LeakyReLU}
     *     activation function
     */
    public static Block leakyReluBlock(float alpha) {
        return new LambdaBlock(arrays -> Activation.leakyRelu(arrays, alpha));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #elu(NDList, float) ELU} activation
     * function in its forward function.
     *
     * @param alpha the slope for the activation
     * @return the {@link LambdaBlock} that applies the {@link #elu(NDList, float) ELU} activation
     *     function
     */
    public static Block eluBlock(float alpha) {
        return new LambdaBlock(arrays -> Activation.elu(arrays, alpha));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #selu(NDList) SELU} activation function
     * in its forward function.
     *
     * @return the {@link LambdaBlock} that applies the {@link #selu(NDList) SELU} activation
     *     function
     */
    public static Block seluBlock() {
        return new LambdaBlock(Activation::selu);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #gelu(NDList) GELU} activation function
     * in its forward function.
     *
     * @return the {@link LambdaBlock} that applies the {@link #gelu(NDList) GELU} activation
     *     function
     */
    public static Block geluBlock() {
        return new LambdaBlock(Activation::gelu);
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #swish(NDList, float) Swish} activation
     * function in its forward function.
     *
     * @param beta a hyper-parameter
     * @return the {@link LambdaBlock} that applies the {@link #swish(NDList, float) Swish}
     *     activation function
     */
    public static Block swishBlock(float beta) {
        return new LambdaBlock(arrays -> Activation.swish(arrays, beta));
    }

    /**
     * Creates a {@link LambdaBlock} that applies the {@link #mish(NDList) Mish} activation function
     * in its forward function.
     *
     * @return the {@link LambdaBlock} that applies the {@link #mish(NDList) Mish} activation
     *     function
     */
    public static Block mishBlock() {
        return new LambdaBlock(Activation::mish);
    }

    /**
     * Returns a {@link Prelu} block.
     *
     * @return a {@link Prelu} block
     */
    public static Block preluBlock() {
        return new Prelu();
    }
}

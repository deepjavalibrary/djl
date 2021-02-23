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
package ai.djl.basicmodelzoo.basic;

import ai.djl.ndarray.NDList;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import java.util.function.Function;

/**
 * Multilayer Perceptron (MLP) NeuralNetworks.
 *
 * <p>A multilayer perceptron (MLP) is a feedforward artificial neural network that generates a set
 * of outputs from a set of inputs. An MLP is characterized by several layers of input nodes
 * connected as a directed graph between the input and output layers. MLP uses backpropogation for
 * training the network.
 *
 * <p>MLP is widely used for solving problems that require supervised learning as well as research
 * into computational neuroscience and parallel distributed processing. Applications include speech
 * recognition, image recognition and machine translation.
 *
 * @see <a href="https://d2l.djl.ai/chapter_multilayer-perceptrons/mlp.html">The D2L chapters on
 *     MLPs</a>
 */
public class Mlp extends SequentialBlock {

    /**
     * Create an MLP NeuralNetwork using RELU.
     *
     * @param input the size of the input vector
     * @param output the size of the output vector
     * @param hidden the sizes of all of the hidden layers
     */
    public Mlp(int input, int output, int[] hidden) {
        this(input, output, hidden, Activation::relu);
    }

    /**
     * Create an MLP NeuralNetwork.
     *
     * @param input the size of the input vector
     * @param output the size of the output vector
     * @param hidden the sizes of all of the hidden layers
     * @param activation the activation function to use
     */
    public Mlp(int input, int output, int[] hidden, Function<NDList, NDList> activation) {
        add(Blocks.batchFlattenBlock(input));
        for (int hiddenSize : hidden) {
            add(Linear.builder().setUnits(hiddenSize).build());
            add(activation);
        }

        add(Linear.builder().setUnits(output).build());
    }
}

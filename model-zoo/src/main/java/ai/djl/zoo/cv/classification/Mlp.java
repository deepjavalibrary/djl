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
package ai.djl.zoo.cv.classification;

import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;

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
 */
public class Mlp extends SequentialBlock {

    /**
     * Create a MLP NeuralNetwork.
     *
     * <p>The existing MLP only contains three {@link Linear} blocks with output channels as 128, 64
     * and 10. Users can specify the width and height on the first layer to match the input
     *
     * @param width the width of the input
     * @param height the height of the input
     */
    public Mlp(int width, int height) {
        add(Blocks.batchFlattenBlock(width * (long) height))
                .add(new Linear.Builder().setOutChannels(128).build())
                .add(Activation.reluBlock())
                .add(new Linear.Builder().setOutChannels(64).build())
                .add(Activation.reluBlock())
                .add(new Linear.Builder().setOutChannels(10).build());
    }
}

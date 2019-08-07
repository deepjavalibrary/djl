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
package software.amazon.ai.nn;

import java.util.Collection;
import software.amazon.ai.Parameter;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.convolutional.Conv1D;
import software.amazon.ai.nn.convolutional.Conv2D;
import software.amazon.ai.nn.convolutional.Conv3D;
import software.amazon.ai.nn.core.Embedding;
import software.amazon.ai.nn.core.Linear;
import software.amazon.ai.nn.core.Prelu;
import software.amazon.ai.nn.norm.BatchNorm;
import software.amazon.ai.nn.norm.Dropout;
import software.amazon.ai.nn.recurrent.GRU;
import software.amazon.ai.nn.recurrent.LSTM;
import software.amazon.ai.nn.recurrent.RNN;
import software.amazon.ai.training.optimizer.Sgd;
import software.amazon.ai.training.optimizer.lrscheduler.LrScheduler;
import software.amazon.ai.util.PairList;

/**
 * An internal mapping to Engine specific implementations of Neural Network {@link
 * software.amazon.ai.Block}s.
 */
public abstract class NNIndex {

    ////////////////////////////////////////
    // Blocks
    ////////////////////////////////////////

    public abstract Linear linear(long units, boolean bias);

    public abstract BatchNorm batchNorm2D(int axis, float epsilon, float momentum);

    public abstract Dropout dropout(float probability, int[] sharedAxes);

    public abstract <T> Embedding<T> embedding(
            Collection<T> items, int embeddingSize, boolean useDefault, DataType dataType);

    public abstract Prelu prelu();

    public abstract Conv1D conv1D(
            Shape kernel,
            Shape stride,
            Shape pad,
            Shape dilate,
            int numFilters,
            int numGroups,
            boolean includeBias);

    public abstract Conv2D conv2D(
            Shape kernel,
            Shape stride,
            Shape pad,
            Shape dilate,
            int numFilters,
            int numGroups,
            boolean includeBias);

    public abstract Conv3D conv3D(
            Shape kernel,
            Shape stride,
            Shape pad,
            Shape dilate,
            int numFilters,
            int numGroups,
            boolean includeBias);

    public abstract RNN rnn(
            long stateSize,
            float dropRate,
            int numStackedLayers,
            RNN.Activation activation,
            boolean useSequenceLength,
            boolean useBidirectional,
            boolean stateOutputs);

    public abstract LSTM lstm(
            long stateSize,
            float dropRate,
            int numStackedLayers,
            double lstmStateClipMin,
            double lstmStateClipMax,
            boolean useSequenceLength,
            boolean useBidirectional,
            boolean stateOutputs,
            boolean clipLstmState);

    public abstract GRU gru(
            long stateSize,
            float dropRate,
            int numStackedLayers,
            boolean useSequenceLength,
            boolean useBidirectional,
            boolean stateOutputs);

    ////////////////////////////////////////
    // Optimizers
    ////////////////////////////////////////

    public abstract Sgd sgd(
            PairList<String, Parameter> parameters,
            float rescaleGrad,
            float weightDecays,
            float clipGrad,
            LrScheduler lrScheduler,
            int beginNumUpdate,
            float momentum,
            boolean lazyUpdate);
}

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

import java.util.Collection;
import org.apache.mxnet.engine.optimizer.MxNag;
import org.apache.mxnet.engine.optimizer.MxSgd;
import org.apache.mxnet.nn.convolutional.MxConv1D;
import org.apache.mxnet.nn.convolutional.MxConv2D;
import org.apache.mxnet.nn.convolutional.MxConv3D;
import org.apache.mxnet.nn.core.MxEmbedding;
import org.apache.mxnet.nn.core.MxLinear;
import org.apache.mxnet.nn.core.MxPrelu;
import org.apache.mxnet.nn.norm.MxBatchNorm;
import org.apache.mxnet.nn.norm.MxDropout;
import org.apache.mxnet.nn.recurrent.MxGRU;
import org.apache.mxnet.nn.recurrent.MxLSTM;
import org.apache.mxnet.nn.recurrent.MxRNN;
import software.amazon.ai.Parameter;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.NNIndex;
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
import software.amazon.ai.training.optimizer.learningrate.LrTracker;
import software.amazon.ai.util.PairList;

public class MxNNIndex extends NNIndex {

    ////////////////////////////////////////
    // Blocks
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    public Linear linear(long outChannels, boolean bias) {
        return new MxLinear(outChannels, bias);
    }

    /** {@inheritDoc} */
    @Override
    public BatchNorm batchNorm2D(int axis, float epsilon, float momentum) {
        return new MxBatchNorm(axis, epsilon, momentum);
    }

    /** {@inheritDoc} */
    @Override
    public Dropout dropout(float probability, int[] sharedAxes) {
        return new MxDropout(probability, sharedAxes);
    }

    /** {@inheritDoc} */
    @Override
    public <T> Embedding<T> embedding(
            Collection<T> items, int embeddingSize, boolean useDefault, DataType dataType) {
        return new MxEmbedding<>(items, embeddingSize, useDefault, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public Prelu prelu() {
        return new MxPrelu();
    }

    /** {@inheritDoc} */
    @Override
    public Conv1D conv1D(
            Shape kernel,
            Shape stride,
            Shape pad,
            Shape dilate,
            int numFilters,
            int numGroups,
            boolean includeBias) {
        return new MxConv1D(kernel, stride, pad, dilate, numFilters, numGroups, includeBias);
    }

    /** {@inheritDoc} */
    @Override
    public Conv2D conv2D(
            Shape kernel,
            Shape stride,
            Shape pad,
            Shape dilate,
            int numFilters,
            int numGroups,
            boolean includeBias) {
        return new MxConv2D(kernel, stride, pad, dilate, numFilters, numGroups, includeBias);
    }

    /** {@inheritDoc} */
    @Override
    public Conv3D conv3D(
            Shape kernel,
            Shape stride,
            Shape pad,
            Shape dilate,
            int numFilters,
            int numGroups,
            boolean includeBias) {
        return new MxConv3D(kernel, stride, pad, dilate, numFilters, numGroups, includeBias);
    }

    /** {@inheritDoc} */
    @Override
    public RNN rnn(
            long stateSize,
            float dropRate,
            int numStackedLayers,
            RNN.Activation activation,
            boolean useSequenceLength,
            boolean useBidirectional,
            boolean stateOutputs) {
        return new MxRNN(
                stateSize,
                dropRate,
                numStackedLayers,
                activation,
                useSequenceLength,
                useBidirectional,
                stateOutputs);
    }

    /** {@inheritDoc} */
    @Override
    public LSTM lstm(
            long stateSize,
            float dropRate,
            int numStackedLayers,
            double lstmStateClipMin,
            double lstmStateClipMax,
            boolean useSequenceLength,
            boolean useBidirectional,
            boolean stateOutputs,
            boolean clipLstmState) {
        return new MxLSTM(
                stateSize,
                dropRate,
                numStackedLayers,
                lstmStateClipMin,
                lstmStateClipMax,
                useSequenceLength,
                useBidirectional,
                stateOutputs,
                clipLstmState);
    }

    /** {@inheritDoc} */
    @Override
    public GRU gru(
            long stateSize,
            float dropRate,
            int numStackedLayers,
            boolean useSequenceLength,
            boolean useBidirectional,
            boolean stateOutputs) {
        return new MxGRU(
                stateSize,
                dropRate,
                numStackedLayers,
                useSequenceLength,
                useBidirectional,
                stateOutputs);
    }

    ////////////////////////////////////////
    // Optimizers
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    public Sgd sgd(
            PairList<String, Parameter> parameters,
            float rescaleGrad,
            float weightDecays,
            float clipGrad,
            LrTracker lrTracker,
            int beginNumUpdate,
            float momentum,
            boolean lazyUpdate) {
        return new MxSgd(
                parameters,
                rescaleGrad,
                weightDecays,
                clipGrad,
                lrTracker,
                beginNumUpdate,
                momentum,
                lazyUpdate);
    }

    @Override
    public MxNag nag(
            PairList<String, Parameter> parameters,
            float rescaleGrad,
            float weightDecays,
            float clipGrad,
            LrTracker lrTracker,
            int beginNumUpdate,
            float momentum) {
        return new MxNag(
                parameters,
                rescaleGrad,
                weightDecays,
                clipGrad,
                lrTracker,
                beginNumUpdate,
                momentum);
    }
}

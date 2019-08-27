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

import org.apache.mxnet.engine.optimizer.MxAdam;
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
import software.amazon.ai.training.optimizer.Adam;
import software.amazon.ai.training.optimizer.Adam.Builder;
import software.amazon.ai.training.optimizer.Nag;
import software.amazon.ai.training.optimizer.Sgd;

public class MxNNIndex extends NNIndex {

    ////////////////////////////////////////
    // Blocks
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    public Linear linear(Linear.Builder builder) {
        return new MxLinear(builder);
    }

    /** {@inheritDoc} */
    @Override
    public BatchNorm batchNorm2D(BatchNorm.Builder builder) {
        return new MxBatchNorm(builder);
    }

    /** {@inheritDoc} */
    @Override
    public Dropout dropout(Dropout.Builder builder) {
        return new MxDropout(builder);
    }

    /** {@inheritDoc} */
    @Override
    public <T> Embedding<T> embedding(Embedding.Builder<T> builder) {
        return new MxEmbedding<>(builder);
    }

    /** {@inheritDoc} */
    @Override
    public Prelu prelu(Prelu.Builder builder) {
        return new MxPrelu();
    }

    /** {@inheritDoc} */
    @Override
    public Conv1D conv1D(Conv1D.Builder builder) {
        return new MxConv1D(builder);
    }

    /** {@inheritDoc} */
    @Override
    public Conv2D conv2D(Conv2D.Builder builder) {
        return new MxConv2D(builder);
    }

    /** {@inheritDoc} */
    @Override
    public Conv3D conv3D(Conv3D.Builder builder) {
        return new MxConv3D(builder);
    }

    /** {@inheritDoc} */
    @Override
    public RNN rnn(RNN.Builder builder) {
        return new MxRNN(builder);
    }

    /** {@inheritDoc} */
    @Override
    public LSTM lstm(LSTM.Builder builder) {
        return new MxLSTM(builder);
    }

    /** {@inheritDoc} */
    @Override
    public GRU gru(GRU.Builder builder) {
        return new MxGRU(builder);
    }

    ////////////////////////////////////////
    // Optimizers
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    public Sgd sgd(Sgd.Builder builder) {
        return new MxSgd(builder);
    }

    /** {@inheritDoc} */
    @Override
    public Nag nag(Nag.Builder builder) {
        return new MxNag(builder);
    }

    /** {@inheritDoc} */
    @Override
    public Adam adam(Builder builder) {
        return new MxAdam(builder);
    }
}

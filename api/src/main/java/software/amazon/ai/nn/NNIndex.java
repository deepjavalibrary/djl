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
import software.amazon.ai.training.optimizer.Nag;
import software.amazon.ai.training.optimizer.Sgd;

/** An internal mapping to Engine specific implementations of Neural Network {@link Block}s. */
public abstract class NNIndex {

    ////////////////////////////////////////
    // Blocks
    ////////////////////////////////////////

    public abstract Linear linear(Linear.Builder builder);

    public abstract BatchNorm batchNorm2D(BatchNorm.Builder builder);

    public abstract Dropout dropout(Dropout.Builder builder);

    public abstract <T> Embedding<T> embedding(Embedding.Builder<T> builder);

    public abstract Prelu prelu(Prelu.Builder builder);

    public abstract Conv1D conv1D(Conv1D.Builder builder);

    public abstract Conv2D conv2D(Conv2D.Builder builder);

    public abstract Conv3D conv3D(Conv3D.Builder builder);

    public abstract RNN rnn(RNN.Builder builder);

    public abstract LSTM lstm(LSTM.Builder builder);

    public abstract GRU gru(GRU.Builder builder);

    ////////////////////////////////////////
    // Optimizers
    ////////////////////////////////////////

    public abstract Sgd sgd(Sgd.Builder builder);

    public abstract Nag nag(Nag.Builder builder);

    public abstract Adam adam(Adam.Builder builder);
}

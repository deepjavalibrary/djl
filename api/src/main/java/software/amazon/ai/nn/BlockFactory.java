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

import java.util.List;
import java.util.function.Function;
import software.amazon.ai.ndarray.NDList;
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
import software.amazon.ai.training.Activation;
import software.amazon.ai.training.optimizer.Adam;
import software.amazon.ai.training.optimizer.Nag;
import software.amazon.ai.training.optimizer.Sgd;

/** An internal to create Neural Network {@link Block}s. */
public interface BlockFactory {

    Activation activation();

    Block createIdentityBlock();

    SequentialBlock createSequential();

    ParallelBlock createParallel(Function<List<NDList>, NDList> function);

    ParallelBlock createParallel(Function<List<NDList>, NDList> function, List<Block> blocks);

    LambdaBlock createLambda(Function<NDList, NDList> lambda);

    Linear createLinear(Linear.Builder builder);

    BatchNorm createBatchNorm2D(BatchNorm.Builder builder);

    Dropout createDropout(Dropout.Builder builder);

    <T> Embedding<T> createEmbedding(Embedding.Builder<T> builder);

    Prelu createPrelu();

    Conv1D createConv1D(Conv1D.Builder builder);

    Conv2D createConv2D(Conv2D.Builder builder);

    Conv3D createConv3D(Conv3D.Builder builder);

    RNN createRnn(RNN.Builder builder);

    LSTM createLstm(LSTM.Builder builder);

    GRU createGru(GRU.Builder builder);

    Sgd createSgd(Sgd.Builder builder);

    Nag createNag(Nag.Builder builder);

    Adam createAdam(Adam.Builder builder);
}

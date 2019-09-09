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

import java.util.List;
import java.util.function.Function;
import org.apache.mxnet.engine.MxNDManager;
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
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.nn.Block;
import software.amazon.ai.nn.BlockFactory;
import software.amazon.ai.nn.LambdaBlock;
import software.amazon.ai.nn.ParallelBlock;
import software.amazon.ai.nn.SequentialBlock;
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
import software.amazon.ai.training.optimizer.Adam.Builder;
import software.amazon.ai.training.optimizer.Nag;
import software.amazon.ai.training.optimizer.Sgd;

@SuppressWarnings("PMD.CouplingBetweenObjects")
public class MxBlockFactory implements BlockFactory {

    private MxNDManager manager;

    public MxBlockFactory(MxNDManager manager) {
        this.manager = manager;
    }

    public MxNDManager getNDManager() {
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public Activation activation() {
        return new MxActivation(manager);
    }

    /** {@inheritDoc} */
    @Override
    public Block createIdentityBlock() {
        return new LambdaBlock(manager, x -> x);
    }

    /** {@inheritDoc} */
    @Override
    public SequentialBlock createSequential() {
        return new SequentialBlock(manager);
    }

    /** {@inheritDoc} */
    @Override
    public ParallelBlock createParallel(Function<List<NDList>, NDList> function) {
        return new ParallelBlock(manager, function);
    }

    /** {@inheritDoc} */
    @Override
    public ParallelBlock createParallel(
            Function<List<NDList>, NDList> function, List<Block> blocks) {
        return new ParallelBlock(manager, function, blocks);
    }

    /** {@inheritDoc} */
    @Override
    public LambdaBlock createLambda(Function<NDList, NDList> lambda) {
        return new LambdaBlock(manager, lambda);
    }

    /** {@inheritDoc} */
    @Override
    public Linear createLinear(Linear.Builder builder) {
        return new MxLinear(manager, builder);
    }

    /** {@inheritDoc} */
    @Override
    public BatchNorm createBatchNorm2D(BatchNorm.Builder builder) {
        return new MxBatchNorm(manager, builder);
    }

    /** {@inheritDoc} */
    @Override
    public Dropout createDropout(Dropout.Builder builder) {
        return new MxDropout(manager, builder);
    }

    /** {@inheritDoc} */
    @Override
    public <T> Embedding<T> createEmbedding(Embedding.Builder<T> builder) {
        return new MxEmbedding<>(manager, builder);
    }

    /** {@inheritDoc} */
    @Override
    public Prelu createPrelu() {
        return new MxPrelu(manager);
    }

    /** {@inheritDoc} */
    @Override
    public Conv1D createConv1D(Conv1D.Builder builder) {
        return new MxConv1D(manager, builder);
    }

    /** {@inheritDoc} */
    @Override
    public Conv2D createConv2D(Conv2D.Builder builder) {
        return new MxConv2D(manager, builder);
    }

    /** {@inheritDoc} */
    @Override
    public Conv3D createConv3D(Conv3D.Builder builder) {
        return new MxConv3D(manager, builder);
    }

    /** {@inheritDoc} */
    @Override
    public RNN createRnn(RNN.Builder builder) {
        return new MxRNN(manager, builder);
    }

    /** {@inheritDoc} */
    @Override
    public LSTM createLstm(LSTM.Builder builder) {
        return new MxLSTM(manager, builder);
    }

    /** {@inheritDoc} */
    @Override
    public GRU createGru(GRU.Builder builder) {
        return new MxGRU(manager, builder);
    }

    ////////////////////////////////////////
    // Optimizers
    ////////////////////////////////////////

    /** {@inheritDoc} */
    @Override
    public Sgd createSgd(Sgd.Builder builder) {
        return new MxSgd(builder);
    }

    /** {@inheritDoc} */
    @Override
    public Nag createNag(Nag.Builder builder) {
        return new MxNag(builder);
    }

    /** {@inheritDoc} */
    @Override
    public Adam createAdam(Builder builder) {
        return new MxAdam(builder);
    }
}

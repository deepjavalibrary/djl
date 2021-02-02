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
package ai.djl.basicmodelzoo.nlp;

import ai.djl.modality.nlp.Encoder;
import ai.djl.modality.nlp.embedding.TrainableTextEmbedding;
import ai.djl.ndarray.NDList;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.recurrent.RecurrentBlock;

/**
 * {@code SimpleTextEncoder} implements a {@link Encoder} that employs a {@link RecurrentBlock} to
 * encode text input.
 */
public class SimpleTextEncoder extends Encoder {

    private static final byte VERSION = 1;

    /**
     * Contructs a new instance of {@code SimpleTextEncoder} with the given {@link RecurrentBlock}.
     * Use this constructor if you are planning to use pre-trained embeddings that don't need
     * further training.
     *
     * @param recurrentBlock the recurrent block to be used to encode
     */
    public SimpleTextEncoder(RecurrentBlock recurrentBlock) {
        super(VERSION, recurrentBlock);
    }

    /**
     * Contructs a new instance of {@code SimpleTextEncoder} with the given {@link RecurrentBlock}
     * and {@link TrainableTextEmbedding}. Use this constructor if you are planning to use
     * pre-trained or fresh embeddings that need further training.
     *
     * @param trainableTextEmbedding the {@link TrainableTextEmbedding} to train embeddings with
     * @param recurrentBlock the recurrent block to be used to encode
     */
    public SimpleTextEncoder(
            TrainableTextEmbedding trainableTextEmbedding, RecurrentBlock recurrentBlock) {
        super(VERSION, new SequentialBlock().add(trainableTextEmbedding).add(recurrentBlock));
    }

    /** {@inheritDoc} */
    @Override
    public NDList getStates(NDList encoderOutput) {
        NDList ret = new NDList(encoderOutput.get(1));
        if (encoderOutput.size() == 3) {
            ret.add(encoderOutput.get(2));
        }
        return ret;
    }
}

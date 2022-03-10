/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.fasttext.zoo.nlp.word_embedding;

import ai.djl.fasttext.FtAbstractBlock;
import ai.djl.fasttext.jni.FtWrapper;
import ai.djl.ndarray.NDList;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import ai.djl.util.passthrough.PassthroughNDArray;

/** A {@link FtAbstractBlock} for {@link ai.djl.Application.NLP#WORD_EMBEDDING}. */
public class FtWordEmbeddingBlock extends FtAbstractBlock {

    /**
     * Constructs a {@link FtWordEmbeddingBlock}.
     *
     * @param fta the {@link FtWrapper} for the "fasttext model".
     */
    public FtWordEmbeddingBlock(FtWrapper fta) {
        super(fta);
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        PassthroughNDArray inputWrapper = (PassthroughNDArray) inputs.singletonOrThrow();
        String input = (String) inputWrapper.getObject();
        float[] result = embedWord(input);
        return new NDList(new PassthroughNDArray(result));
    }
}

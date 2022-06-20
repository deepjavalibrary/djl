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
package ai.djl.fasttext;

import ai.djl.fasttext.jni.FtWrapper;
import ai.djl.nn.AbstractSymbolBlock;
import ai.djl.nn.ParameterList;

import java.nio.file.Path;

/**
 * A parent class containing shared behavior for {@link ai.djl.nn.SymbolBlock}s based on fasttext
 * models.
 */
public abstract class FtAbstractBlock extends AbstractSymbolBlock implements AutoCloseable {

    protected FtWrapper fta;

    protected Path modelFile;

    /**
     * Constructs a {@link FtAbstractBlock}.
     *
     * @param fta the {@link FtWrapper} containing the "fasttext model"
     */
    public FtAbstractBlock(FtWrapper fta) {
        this.fta = fta;
    }

    /**
     * Returns the fasttext model file for the block.
     *
     * @return the fasttext model file for the block
     */
    public Path getModelFile() {
        return modelFile;
    }

    /**
     * Embeds a word using fasttext.
     *
     * @param word the word to embed
     * @return the embedding
     * @see ai.djl.modality.nlp.embedding.WordEmbedding
     */
    public float[] embedWord(String word) {
        return fta.getWordVector(word);
    }

    /** {@inheritDoc} */
    @Override
    public ParameterList getDirectParameters() {
        throw new UnsupportedOperationException("Not yet supported");
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        fta.unloadModel();
        fta.close();
    }
}

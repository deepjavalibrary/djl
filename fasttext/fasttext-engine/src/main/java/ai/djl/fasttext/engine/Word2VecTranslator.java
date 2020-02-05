/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.fasttext.engine;

import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import com.github.jfasttext.FastTextWrapper;

/** A word2vec @{link Translator} for fastText. */
public class Word2VecTranslator implements Translator<String, float[]> {

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, String input) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public float[] processOutput(TranslatorContext ctx, NDList list) {
        String input = (String) ctx.getAttachment("input");
        FtModel model = (FtModel) ctx.getModel();
        FastTextWrapper.RealVector rv = model.fta.getVector(input);

        int size = (int) rv.size();
        float[] vec = new float[size];
        for (int i = 0; i < size; ++i) {
            vec[i] = rv.get(i);
        }
        return vec;
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return null;
    }
}

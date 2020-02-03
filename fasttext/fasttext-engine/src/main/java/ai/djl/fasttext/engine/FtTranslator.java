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

import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import com.github.jfasttext.FastTextWrapper;
import java.util.ArrayList;
import java.util.List;

/** A text classification @{link Translator} for fastText. */
public class FtTranslator implements Translator<String, Classifications> {

    private int topK;

    /** Constructs an instance of {@link FtTranslator}. */
    public FtTranslator() {
        this(1);
    }

    /**
     * Constructs an instance of {@link FtTranslator} that return @{code topK} classifications.
     *
     * @param topK the value of K
     */
    public FtTranslator(int topK) {
        this.topK = topK;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, String input) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) {
        String input = (String) ctx.getAttachment("input");
        FtModel model = (FtModel) ctx.getModel();
        FastTextWrapper.FloatStringPairVector fspv = model.fta.predictProba(input, topK);

        int size = Math.min((int) fspv.size(), topK);
        List<String> classNames = new ArrayList<>(size);
        List<Double> probabilities = new ArrayList<>(size);
        for (int i = 0; i < size; ++i) {
            probabilities.add(Math.exp(fspv.first(i)));
            classNames.add(fspv.second(i).getString().substring(9));
        }
        return new Classifications(classNames, probabilities);
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return null;
    }
}

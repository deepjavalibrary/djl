/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.huggingface.translator;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonUtils;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;

/** The translator for Huggingface text classification model. */
public class TextClassificationBatchTranslator
        implements NoBatchifyTranslator<String[], Classifications[]> {

    private HuggingFaceTokenizer tokenizer;
    private boolean includeTokenTypes;
    private Batchifier batchifier;
    private PretrainedConfig config;

    TextClassificationBatchTranslator(
            HuggingFaceTokenizer tokenizer,
            boolean includeTokenTypes,
            Batchifier batchifier,
            PretrainedConfig config) {
        this.tokenizer = tokenizer;
        this.includeTokenTypes = includeTokenTypes;
        this.batchifier = batchifier;
        this.config = config;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
        Path path = ctx.getModel().getModelPath();
        Path file = path.resolve("config.json");
        try (Reader reader = Files.newBufferedReader(file)) {
            config = JsonUtils.GSON.fromJson(reader, PretrainedConfig.class);
        }
    }
    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, String[] inputs) {
        NDManager manager = ctx.getNDManager();
        Encoding[] encodings = tokenizer.batchEncode(inputs);
        NDList[] batch = new NDList[encodings.length];
        for (int i = 0; i < encodings.length; ++i) {
            batch[i] = encodings[i].toNDList(manager, includeTokenTypes);
        }
        return batchifier.batchify(batch);
    }

    /** {@inheritDoc} */
    @Override
    public Classifications[] processOutput(TranslatorContext ctx, NDList list) {
        NDList[] batch = batchifier.unbatchify(list);
        Classifications[] ret = new Classifications[batch.length];
        for (int i = 0; i < batch.length; ++i) {
            ret[i] = TextClassificationTranslator.toClassifications(config, batch[i]);
        }
        return ret;
    }
}

/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.mxnet.zoo.nlp.embedding;

import ai.djl.Model;
import ai.djl.modality.nlp.SimpleVocabulary;
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding;
import ai.djl.nn.Block;
import ai.djl.nn.BlockFactory;
import ai.djl.util.Utils;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/** A {@link BlockFactory} class that creates Glove word embedding block. */
public class GloveWordEmbeddingBlockFactory implements BlockFactory {

    private static final long serialVersionUID = 1L;

    /** {@inheritDoc} */
    @Override
    public Block newBlock(Model model, Path modelPath, Map<String, ?> arguments)
            throws IOException {
        List<String> idxToToken = Utils.readLines(modelPath.resolve("idx_to_token.txt"));
        String dimension = (String) arguments.get("dimensions");
        TrainableWordEmbedding wordEmbedding =
                TrainableWordEmbedding.builder()
                        .optNumEmbeddings(Integer.parseInt(dimension))
                        .setVocabulary(new SimpleVocabulary(idxToToken))
                        .optUnknownToken((String) arguments.get("unknownToken"))
                        .optUseDefault(true)
                        .build();
        model.setProperty("unknownToken", (String) arguments.get("unknownToken"));
        return wordEmbedding;
    }
}

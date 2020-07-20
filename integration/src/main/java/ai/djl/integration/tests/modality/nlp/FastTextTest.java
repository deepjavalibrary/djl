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
package ai.djl.integration.tests.modality.nlp;

import ai.djl.MalformedModelException;
import ai.djl.fasttext.FtModel;
import ai.djl.fasttext.FtVocabulary;
import ai.djl.fasttext.FtWord2VecWordEmbedding;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class FastTextTest {

    @BeforeClass
    public void setup() throws IOException {
        URL url =
                new URL(
                        "https://djl-ai.s3.amazonaws.com/mlrepo/model/nlp/text_classification/ai/djl/fasttext/cooking_stackexchange/0.0.1/cooking.ftz");
        Path path = Paths.get("build/tmp/model");
        Path modelFile = path.resolve("cooking.ftz");
        if (!Files.exists(modelFile)) {
            Files.createDirectories(path);
            try (InputStream is = url.openStream()) {
                Files.copy(is, modelFile);
            }
        }
    }

    @Test
    public void testWord2Vec() throws IOException, MalformedModelException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Path path = Paths.get("build/tmp/model");
            FtModel ftModel = new FtModel("word2vec");
            ftModel.load(path, "cooking");
            FtWord2VecWordEmbedding fasttextWord2VecWordEmbedding =
                    new FtWord2VecWordEmbedding(ftModel, new FtVocabulary());
            long index = fasttextWord2VecWordEmbedding.preprocessWordToEmbed("bread");
            NDArray embedding = fasttextWord2VecWordEmbedding.embedWord(manager, index);
            Assert.assertEquals(embedding.getShape(), new Shape(100));
            Assert.assertEquals(embedding.toFloatArray()[0], 0.038162477, 0.001);
        }
    }
}

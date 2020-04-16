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
package ai.djl.mxnet.integration.modality.nlp;

import ai.djl.MalformedModelException;
import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.modality.nlp.embedding.ModelZooWordEmbedding;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import java.io.IOException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class WordEmbeddingTest {

    @Test
    public void testGlove()
            throws IOException, ModelNotFoundException, MalformedModelException,
                    EmbeddingException {
        try (ZooModel<NDList, NDList> model = MxModelZoo.GLOVE.loadModel()) {
            try (ModelZooWordEmbedding wordEmbedding = new ModelZooWordEmbedding(model)) {
                NDManager manager = model.getNDManager();
                NDList result = new NDList(wordEmbedding.embedWord(manager, "the"));
                Assert.assertEquals(result.singletonOrThrow().sum().getFloat(), -5.24, .01);
            }
        }
    }
}

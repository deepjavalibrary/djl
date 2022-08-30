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
package ai.djl.huggingface.zoo;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.testing.TestRequirements;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class ModelZooTest {

    @Test
    public void testModelZoo() throws ModelException, IOException, TranslateException {
        TestRequirements.nightly();

        String question = "When did BBC Japan start broadcasting?";
        String paragraph =
                "BBC Japan was a general entertainment Channel. "
                        + "Which operated between December 2004 and April 2006. "
                        + "It ceased operations after its Japanese distributor folded.";

        String url = "djl://ai.djl.huggingface.pytorch/deepset/minilm-uncased-squad2";
        Criteria<QAInput, String> criteria =
                Criteria.builder().setTypes(QAInput.class, String.class).optModelUrls(url).build();

        try (ZooModel<QAInput, String> model = criteria.loadModel();
                Predictor<QAInput, String> predictor = model.newPredictor()) {
            QAInput input = new QAInput(question, paragraph);
            String res = predictor.predict(input);
            Assert.assertEquals(res, "december 2004");
        }
    }
}

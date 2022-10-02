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
package ai.djl.tensorflow.integration;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.testing.TestRequirements;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class ModelLoadingTest {

    @Test
    public void loadModelWithStringTensor() throws ModelException, IOException, TranslateException {
        TestRequirements.nightly();
        String url = "https://resources.djl.ai/test-models/tensorflow/string_tensor.zip";
        Criteria<NDList, NDList> criteria =
                Criteria.builder().setTypes(NDList.class, NDList.class).optModelUrls(url).build();

        try (ZooModel<NDList, NDList> model = criteria.loadModel();
                Predictor<NDList, NDList> predictor = model.newPredictor();
                NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.create("Test1");
            NDList output = predictor.predict(new NDList(array));
            Assert.assertEquals(output.size(), 2);
            NDArray str = output.get("corrected_query");
            Assert.assertEquals(str.getDataType(), DataType.STRING);
        }
    }
}

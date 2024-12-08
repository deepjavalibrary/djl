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
package ai.djl.examples.inference.nlp;

import ai.djl.ModelException;
import ai.djl.testing.TestRequirements;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;

import com.google.gson.JsonObject;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class BertQaTest {

    @Test
    public void testBertQa() throws ModelException, TranslateException, IOException {
        TestRequirements.linux();

        String result = BertQaInference.predict();
        JsonObject json = JsonUtils.GSON.fromJson(result, JsonObject.class);
        String answer = json.get("answer").getAsString();
        Assert.assertEquals(answer, "december 2004");
    }
}

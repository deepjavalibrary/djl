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
package ai.djl.examples.inference;

import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.modality.Classifications;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class SentimentAnalysisTest {

    @Test
    public void testSentimentAnalysis() throws ModelException, TranslateException, IOException {
        if (!Boolean.getBoolean("nightly")) {
            throw new SkipException("Nightly only");
        }
        if (!"PyTorch".equals(Engine.getInstance().getEngineName())) {
            throw new SkipException("Only works for PyTorch engine.");
        }

        Classifications result = SentimentAnalysis.predict();
        Assert.assertEquals(result.best().getClassName(), "Positive");
    }
}

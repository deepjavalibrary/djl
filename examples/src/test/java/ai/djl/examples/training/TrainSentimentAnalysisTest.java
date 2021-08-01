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
package ai.djl.examples.training;

import ai.djl.MalformedModelException;
import ai.djl.engine.Engine;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class TrainSentimentAnalysisTest {

    @Test
    public void testTrainSentimentAnalysis()
            throws MalformedModelException, ModelNotFoundException, TranslateException,
                    IOException {
        // this is nightly test
        if (!Boolean.getBoolean("nightly")) {
            throw new SkipException("Nightly only");
        }
        if (Engine.getInstance().getGpuCount() == 0) {
            throw new SkipException("GPU only");
        }

        String[] args = new String[] {"-e", "1", "-g", "1"};
        TrainSentimentAnalysis.runExample(args);
    }
}

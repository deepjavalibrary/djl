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

import ai.djl.ModelException;
import ai.djl.testing.TestRequirements;
import ai.djl.translate.TranslateException;

import org.testng.annotations.Test;

import java.io.IOException;

public class TrainSentimentAnalysisTest {

    @Test
    public void testTrainSentimentAnalysis()
            throws ModelException, TranslateException, IOException {
        TestRequirements.engine("MXNet");
        TestRequirements.nightly();
        TestRequirements.gpu();

        String[] args = new String[] {"-e", "1", "-g", "1"};
        TrainSentimentAnalysis.runExample(args);
    }
}

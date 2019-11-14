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
package ai.djl.examples;

import ai.djl.MalformedModelException;
import ai.djl.engine.Engine;
import ai.djl.examples.training.TrainPikachu;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class TrainPikachuTest {

    @Test
    public void testDetection() throws IOException, MalformedModelException, TranslateException {
        // only test on gpu
        if (Engine.getInstance().getGpuCount() > 0) {
            // test train
            String[] args = {"-e", "20", "-b", "32", "-g", "1", "-o", "build/output"};
            TrainPikachu trainPikachu = new TrainPikachu();
            Assert.assertTrue(trainPikachu.runExample(args));
            Assert.assertTrue(trainPikachu.getValidationLoss() < 2.5e-3);
            // test predict
            int numberOfPikachus =
                    trainPikachu.predict("build/output", "src/test/resources/pikachu.jpg");
            Assert.assertTrue(numberOfPikachus >= 6);
            Assert.assertTrue(numberOfPikachus <= 9);
        }
    }
}

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

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.examples.training.TrainPikachu;
import ai.djl.examples.training.util.ExampleTrainingResult;
import ai.djl.mxnet.jna.JnaUtils;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import org.apache.commons.cli.ParseException;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class TrainPikachuTest {

    @Test
    public void testDetection()
            throws IOException, MalformedModelException, TranslateException, ParseException {
        // this is nightly test
        if (!Boolean.getBoolean("nightly")) {
            throw new SkipException("Nightly only");
        }
        String[] args;
        float expectedLoss = 0;
        int expectedMinNumber = 0;
        int expectedMaxNumber = 0;
        if (Device.getGpuCount() > 0) {
            args = new String[] {"-e", "20", "-b", "32", "-g", "1"};
            expectedLoss = 2.5e-3f;
            expectedMaxNumber = 15;
            expectedMinNumber = 6;
        } else {
            // test train 1 epoch and predict workflow works on CPU
            args = new String[] {"-e", "1", "-m", "1", "-b", "32"};
        }
        // test train
        TrainPikachu trainPikachu = new TrainPikachu();
        ExampleTrainingResult result = trainPikachu.runExample(args);

        if (expectedLoss > 0) {
            Assert.assertTrue(result.getEvaluation("SingleShotDetectionLoss") < expectedLoss);
        }

        JnaUtils.waitAll();
        // test predict
        int numberOfPikachus =
                new TrainPikachu().predict("build/model", "src/test/resources/pikachu.jpg");
        if (expectedMinNumber > 0) {
            Assert.assertTrue(numberOfPikachus >= expectedMinNumber);
            Assert.assertTrue(numberOfPikachus <= expectedMaxNumber);
        }
    }
}

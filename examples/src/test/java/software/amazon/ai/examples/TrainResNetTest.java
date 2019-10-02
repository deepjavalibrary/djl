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

package software.amazon.ai.examples;

import java.io.IOException;
import org.apache.commons.cli.ParseException;
import org.testng.Assert;
import org.testng.annotations.Test;
import software.amazon.ai.engine.Engine;
import software.amazon.ai.examples.training.transferlearning.TrainResnetWithCifar10;
import software.amazon.ai.zoo.ModelNotFoundException;

public class TrainResNetTest {
    @Test
    public void testTrainResNet() throws ParseException, IOException, ModelNotFoundException {
        int numGpus = Engine.getInstance().getGpuCount();
        // only run this test on gpu
        if (numGpus > 0) {
            String[] args =
                    new String[] {
                        "-b",
                        String.valueOf(32 * numGpus),
                        "-e",
                        "5",
                        "-g",
                        String.valueOf(numGpus),
                        "-s",
                        "true",
                        "-p",
                        "true"
                    };
            TrainResnetWithCifar10.main(args);
            Assert.assertTrue(TrainResnetWithCifar10.getAccuracy() > 0.7f);
            Assert.assertTrue(TrainResnetWithCifar10.getLossValue() < 1f);
        }
    }
}

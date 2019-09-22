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
import software.amazon.ai.examples.training.TrainMnist;
import software.amazon.ai.translate.TranslateException;

public class TrainMnistTest {

    @Test
    public void testTrainMnist() throws TranslateException, ParseException, IOException {
        String[] args = {"-e", "2"};
        TrainMnist.main(args);
        Assert.assertTrue(TrainMnist.getAccuracy() > 0.9f);
        Assert.assertTrue(TrainMnist.getLossValue() < 0.2f);
    }
}

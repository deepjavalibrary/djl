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

import ai.djl.testing.TestRequirements;
import ai.djl.training.TrainingResult;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class TrainSeq2SeqTest {

    @Test
    public void testTrainSeq2Seq() throws IOException, TranslateException {
        TestRequirements.notArm();
        TestRequirements.linux();

        // TODO: PyTorch -- PtNDArray.sequenceMask not implemented
        String[] args = new String[] {"-g", "1", "-e", "1", "-m", "2", "--engine", "MXNet"};
        TrainingResult result = TrainSeq2Seq.runExample(args);
        Assert.assertNotNull(result);
    }
}

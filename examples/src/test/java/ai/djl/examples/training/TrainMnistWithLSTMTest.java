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

import ai.djl.engine.Engine;
import ai.djl.training.TrainingResult;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class TrainMnistWithLSTMTest {

    @Test
    public void testTrainMnistWithLSTM() throws IOException, TranslateException {
        String[] args;
        Engine engine = Engine.getEngine("PyTorch");
        if (engine.getGpuCount() > 0) {
            // TODO: PyTorch -- cuDNN error: CUDNN_STATUS_VERSION_MISMATCH
            args = new String[] {"-g", "1", "-e", "1", "-m", "2", "--engine", "MXNet"};
        } else {
            args = new String[] {"-g", "1", "-e", "1", "-m", "2"};
        }
        TrainingResult result = TrainMnistWithLSTM.runExample(args);
        Assert.assertNotNull(result);
    }
}

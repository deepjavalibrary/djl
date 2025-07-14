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
package ai.djl.examples.training;

import ai.djl.testing.TestRequirements;
import ai.djl.training.TrainingResult;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class TrainCaptchaTest {

    @Test
    public void testTrainCaptcha() throws IOException, TranslateException {
        TestRequirements.notArm();
        TestRequirements.linux();

        // TODO: PyTorch
        /*
        ai.djl.engine.EngineException: index 11 is out of bounds for dimension 1 with size 11
            at app//ai.djl.pytorch.jni.PyTorchLibrary.torchGather(Native Method)
            at app//ai.djl.pytorch.jni.JniUtils.pick(JniUtils.java:581)
            at app//ai.djl.pytorch.jni.JniUtils.indexAdv(JniUtils.java:417)
            at app//ai.djl.pytorch.engine.PtNDArrayIndexer.get(PtNDArrayIndexer.java:74)
            at app//ai.djl.ndarray.NDArray.get(NDArray.java:614)
            at app//ai.djl.ndarray.NDArray.get(NDArray.java:603)
            at app//ai.djl.training.loss.SoftmaxCrossEntropyLoss.evaluate(SoftmaxCrossEntropyLoss.java:86)
            at app//ai.djl.training.loss.IndexLoss.evaluate(IndexLoss.java:55)
            at app//ai.djl.training.loss.AbstractCompositeLoss.evaluate(AbstractCompositeLoss.java:68)
            at app//ai.djl.training.EasyTrain.trainSplit(EasyTrain.java:124)
            at app//ai.djl.training.EasyTrain.trainBatch(EasyTrain.java:110)
            at app//ai.djl.training.EasyTrain.fit(EasyTrain.java:58)
         */
        String[] args = new String[] {"-g", "1", "-e", "1", "-m", "2", "--engine", "MXNet"};
        TrainingResult result = TrainCaptcha.runExample(args);
        Assert.assertNotNull(result);
    }
}

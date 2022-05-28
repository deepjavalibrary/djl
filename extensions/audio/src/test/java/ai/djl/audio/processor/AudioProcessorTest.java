/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.audio.processor;

import ai.djl.Device;
import ai.djl.audio.AudioUtils;
import ai.djl.audio.dataset.AudioData;
import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import java.util.Arrays;
import org.testng.Assert;
import org.testng.annotations.Test;

public class AudioProcessorTest {

    private static String filePath = "src/test/resources/test.wav";
    private static float eps = 1e-3f;

    @Test
    public void testAudioNormalizer() throws EmbeddingException {
        NDManager manager = NDManager.newBaseManager(Device.cpu());
        AudioData.Configuration configuration =
                new AudioData.Configuration()
                        .setProcessorList(Arrays.asList(new AudioNormalizer(-20)));
        AudioData testData = new AudioData(configuration);
        testData.setAudioPaths(Arrays.asList(filePath));
        NDArray samples = testData.getPreprocessedData(manager, 0);
        Assert.assertEquals(AudioUtils.rmsDb(samples), -20.0f, 1e-3);
    }

    @Test
    public static void testLinearSpecgram() throws EmbeddingException {
        NDManager manager = NDManager.newBaseManager(Device.cpu());
        AudioData.Configuration configuration =
                new AudioData.Configuration()
                        .setSampleRate(16000)
                        .setProcessorList(
                                Arrays.asList(
                                        new AudioNormalizer(-20),
                                        new LinearSpecgram(10, 20, 16000)));
        AudioData testData = new AudioData(configuration);
        testData.setAudioPaths(Arrays.asList(filePath));
        NDArray samples = testData.getPreprocessedData(manager, 0);
        Assert.assertTrue(samples.getShape().equals(new Shape(161, 838)));
        Assert.assertEquals(samples.get("0,0").toFloatArray()[0], -15.4571f, eps);
    }
}

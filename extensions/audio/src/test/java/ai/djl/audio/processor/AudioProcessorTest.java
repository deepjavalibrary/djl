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
import ai.djl.audio.dataset.AudioData;
import ai.djl.audio.util.AudioUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.util.DownloadUtils;

import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;

public class AudioProcessorTest {

    private static final String URL = "https://resources.djl.ai/audios/";

    @BeforeClass
    public void setUp() throws IOException {
        DownloadUtils.download(URL + "test_01.wav", "build/test/test_01.wav");
        DownloadUtils.download(URL + "mel_80_filters.npz", "build/test/mel_80_filters.npz");
    }

    @Test
    public void testAudioNormalizer() {
        NDManager manager = NDManager.newBaseManager(Device.cpu());
        AudioData.Configuration configuration =
                new AudioData.Configuration()
                        .setProcessorList(Collections.singletonList(new AudioNormalizer(-20)));
        AudioData testData = new AudioData(configuration);
        testData.setAudioPaths(Collections.singletonList("build/test/test_01.wav"));
        NDArray samples = testData.getPreprocessedData(manager, 0);
        Assert.assertEquals(AudioUtils.rmsDb(samples), -20.0f, 1e-3);
    }

    @Test
    public void testPadOrTrim() {
        try (NDManager manager = NDManager.newBaseManager()) {
            AudioProcessor processor = new PadOrTrim(100);
            NDArray test = manager.ones(new Shape(10));
            NDArray result = processor.extractFeatures(manager, test);
            Assert.assertEquals(result.getShape(), new Shape(100));
            test = manager.ones(new Shape(1000));
            result = processor.extractFeatures(manager, test);
            Assert.assertEquals(result.getShape(), new Shape(100));
        }
    }

    @Test
    void testLogMelSpectrogram() throws IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            AudioProcessor processor =
                    new LogMelSpectrogram(Paths.get("build/test/mel_80_filters.npz"), 80, manager);
            NDArray samples = manager.zeros(new Shape(480000));
            NDArray result = processor.extractFeatures(manager, samples);
            Assert.assertEquals(result.getShape(), new Shape(80, 3000));
        }
    }

    @Test
    public static void testLinearSpecgram() {
        NDManager manager = NDManager.newBaseManager(Device.cpu());
        AudioData.Configuration configuration =
                new AudioData.Configuration()
                        .setSampleRate(16000)
                        .setProcessorList(
                                Arrays.asList(
                                        new AudioNormalizer(-20),
                                        new LinearSpecgram(10, 20, 16000)));
        AudioData testData = new AudioData(configuration);
        testData.setAudioPaths(Collections.singletonList("build/test/test_01.wav"));
        NDArray samples = testData.getPreprocessedData(manager, 0);
        Assert.assertEquals(new Shape(161, 838), samples.getShape());
        Assert.assertEquals(samples.get("0,0").toFloatArray()[0], -15.4531f, 1e-3f);
    }
}

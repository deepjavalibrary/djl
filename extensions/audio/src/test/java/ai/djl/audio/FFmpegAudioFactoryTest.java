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
package ai.djl.audio;

import ai.djl.modality.audio.Audio;
import ai.djl.modality.audio.AudioFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.util.DownloadUtils;

import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Paths;

public class FFmpegAudioFactoryTest {

    private static final String URL = "https://resources.djl.ai/audios/test_01.wav";

    @BeforeClass
    public void setUp() throws IOException {
        DownloadUtils.download(URL, "build/test/test_01.wav");
    }

    @Test
    public void testFactory() throws IOException {
        Audio audio = new FFmpegAudioFactory(null).fromFile(Paths.get("build/test/test_01.wav"));
        Assert.assertEquals(audio.getSampleRate(), 16000f);
        Assert.assertEquals(audio.getChannels(), 1);

        audio =
                new FFmpegAudioFactory(new AudioFactory.Configuration().setSampleRate(10000))
                        .fromUrl("build/test/test_01.wav");
        Assert.assertEquals(audio.getSampleRate(), 10000f);
        Assert.assertEquals(audio.getChannels(), 1);

        audio = new FFmpegAudioFactory(null).fromUrl(URL);
        Assert.assertEquals(audio.getSampleRate(), 16000f);
        Assert.assertEquals(audio.getChannels(), 1);

        float[] data = {0.001f, 0.002f, 0.003f};
        audio = new FFmpegAudioFactory(null).fromData(data);
        Assert.assertEquals(audio.getData(), data);
        Assert.assertEquals(audio.getSampleRate(), 0);
        Assert.assertEquals(audio.getChannels(), 0);
    }

    @Test(expectedExceptions = UnsupportedOperationException.class)
    public void testFromNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.zeros(new Shape(1));
            new FFmpegAudioFactory(null).fromNDArray(array);
        }
    }
}

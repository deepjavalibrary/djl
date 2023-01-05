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
import ai.djl.training.util.DownloadUtils;

import org.bytedeco.ffmpeg.global.avutil;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Paths;

public class FFmpegAudioFactoryTest {

    @BeforeClass
    public void setUp() throws IOException {
        DownloadUtils.download(
                "https://resources.djl.ai/audios/test_01.wav", "build/test/test_01.wav");
    }

    @Test
    public void testFactory() throws IOException {
        AudioFactory factory =
                AudioFactory.newInstance()
                        .setChannels(1)
                        .setSampleRate(10000)
                        .setSampleFormat(avutil.AV_SAMPLE_FMT_S16);

        Assert.assertEquals(factory.getChannels(), 1);
        Assert.assertEquals(factory.getSampleRate(), 10000);
        Assert.assertEquals(factory.getSampleFormat(), avutil.AV_SAMPLE_FMT_S16);

        Audio audio = factory.fromUrl("build/test/test_01.wav");
        Assert.assertEquals(audio.getSampleRate(), 10000f);
        Assert.assertEquals(audio.getChannels(), 1);

        URL url = Paths.get("build/test/test_01.wav").toAbsolutePath().toUri().toURL();

        audio =
                AudioFactory.newInstance()
                        .setSampleRate(8000)
                        .setSampleFormat(avutil.AV_SAMPLE_FMT_S32)
                        .fromUrl(url);
        Assert.assertEquals(audio.getSampleRate(), 8000f);
        Assert.assertEquals(audio.getChannels(), 1);

        Assert.assertThrows(
                () -> {
                    AudioFactory.newInstance()
                            .setSampleFormat(avutil.AV_SAMPLE_FMT_DBL)
                            .fromUrl(url);
                });
    }
}

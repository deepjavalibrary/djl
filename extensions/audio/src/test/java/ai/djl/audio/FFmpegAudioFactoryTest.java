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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
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

    @Test
    public void testPcm16Normalization() throws IOException {
        Path path = Paths.get("build/test/pcm16_extremes.wav");
        Files.createDirectories(path.getParent());
        writeMonoPcm16Wav(path, new short[] {Short.MIN_VALUE, Short.MAX_VALUE, 0});

        Audio audio =
                AudioFactory.newInstance()
                        .setChannels(1)
                        .setSampleRate(16000)
                        .setSampleFormat(avutil.AV_SAMPLE_FMT_S16)
                        .fromFile(path);
        float[] data = audio.getData();
        Assert.assertEquals(data.length, 3);
        Assert.assertEquals(data[0], -1.0f, 0.0f);
        Assert.assertEquals(data[1], Short.MAX_VALUE / 32768.0f, 0.0f);
        Assert.assertEquals(data[2], 0.0f, 0.0f);
    }

    private static void writeMonoPcm16Wav(Path path, short[] samples) throws IOException {
        int dataSize = samples.length * 2;
        ByteBuffer buf = ByteBuffer.allocate(44 + dataSize).order(ByteOrder.LITTLE_ENDIAN);
        buf.put("RIFF".getBytes());
        buf.putInt(36 + dataSize);
        buf.put("WAVE".getBytes());
        buf.put("fmt ".getBytes());
        buf.putInt(16);
        buf.putShort((short) 1);
        buf.putShort((short) 1);
        buf.putInt(16000);
        buf.putInt(32000);
        buf.putShort((short) 2);
        buf.putShort((short) 16);
        buf.put("data".getBytes());
        buf.putInt(dataSize);
        for (short sample : samples) {
            buf.putShort(sample);
        }
        Files.write(path, buf.array());
    }
}

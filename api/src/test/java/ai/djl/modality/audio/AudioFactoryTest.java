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
package ai.djl.modality.audio;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.util.DownloadUtils;

import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;

public class AudioFactoryTest {

    private static final String URL = "https://resources.djl.ai/audios/test_01.wav";

    @BeforeClass
    public void setUp() throws IOException {
        DownloadUtils.download(URL, "build/test/test_01.wav");
    }

    @Test
    public void testFromFile() throws IOException {
        Audio audio = AudioFactory.getInstance().fromFile(Paths.get("build/test/test_01.wav"));
        Assert.assertEquals(audio.getSampleRate(), 16000f);
        Assert.assertEquals(audio.getChannels(), 1);
    }

    @Test
    public void testFromUrl() throws IOException {
        Audio audio = AudioFactory.getInstance().fromUrl("build/test/test_01.wav");
        Assert.assertEquals(audio.getSampleRate(), 16000f);
        Assert.assertEquals(audio.getChannels(), 1);
    }

    @Test
    public void testFromInputStream() throws IOException {
        try (InputStream is = Files.newInputStream(Paths.get("build/test/test_01.wav"))) {
            Audio audio = AudioFactory.getInstance().fromInputStream(is);
            Assert.assertEquals(audio.getSampleRate(), 16000f);
            Assert.assertEquals(audio.getChannels(), 1);
        }
    }

    @Test
    public void testFromData() {
        float[] data = {0.001f, 0.002f, 0.003f};
        Audio audio = AudioFactory.getInstance().fromData(data);
        Assert.assertEquals(audio.getData(), data);
        Assert.assertEquals(audio.getSampleRate(), 0);
        Assert.assertEquals(audio.getChannels(), 0);
    }

    @Test(expectedExceptions = UnsupportedOperationException.class)
    public void testFromNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.zeros(new Shape(1));
            AudioFactory.getInstance().fromNDArray(array);
        }
    }
}

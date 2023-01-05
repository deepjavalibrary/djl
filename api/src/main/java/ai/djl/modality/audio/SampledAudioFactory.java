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

import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;
import java.nio.file.Path;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;

/**
 * {@code SampledAudioFactory} is an implementation of {@link ImageFactory} using the Java Sampled
 * Package.
 *
 * @see <a
 *     href="http://google.com">https://docs.oracle.com/javase/tutorial/sound/sampled-overview.html</a>
 */
public class SampledAudioFactory extends AudioFactory {

    /**
     * Simple Audio Factory implementation.
     *
     * @param configuration configuration to pass
     */
    public SampledAudioFactory(Configuration configuration) {
        super(configuration);
        if (configuration != null) {
            throw new UnsupportedOperationException(
                    "Configuration not supported for default Audio Factory");
        }
    }

    /** {@inheritDoc} */
    @Override
    public Audio fromFile(Path path) throws IOException {
        try (AudioInputStream ais = AudioSystem.getAudioInputStream(path.toFile())) {
            AudioFormat format = ais.getFormat();
            byte[] bytes = read(ais);
            float[] floats =
                    bytesToFloats(
                            bytes,
                            format.isBigEndian() ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN);
            return new Audio(floats, format.getSampleRate(), format.getChannels());
        } catch (UnsupportedAudioFileException e) {
            throw new IOException("Unsupported Audio file", e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public Audio fromInputStream(InputStream is) throws IOException {
        try (AudioInputStream ais = AudioSystem.getAudioInputStream(new BufferedInputStream(is))) {
            AudioFormat format = ais.getFormat();
            byte[] bytes = read(ais);
            float[] floats =
                    bytesToFloats(
                            bytes,
                            format.isBigEndian() ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN);
            return new Audio(floats, format.getSampleRate(), format.getChannels());
        } catch (UnsupportedAudioFileException e) {
            throw new IOException("Unsupported Audio file", e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public Audio fromData(float[] data) {
        return new Audio(data);
    }

    /** {@inheritDoc} */
    @Override
    public Audio fromNDArray(NDArray array) {
        throw new UnsupportedOperationException("Not supported!");
    }

    private byte[] read(AudioInputStream ais) throws IOException {
        AudioFormat format = ais.getFormat();
        int frameSize = format.getFrameSize();

        // Some audio formats may have unspecified frame size
        if (frameSize == AudioSystem.NOT_SPECIFIED) {
            frameSize = 1;
        }

        int size = (int) ais.getFrameLength() * frameSize;
        byte[] ret = new byte[size];
        byte[] buf = new byte[1024];
        int offset = 0;
        int read;
        while ((read = ais.read(buf)) != -1) {
            System.arraycopy(buf, 0, ret, offset, read);
            offset += read;
        }
        return ret;
    }

    private float[] bytesToFloats(byte[] bytes, ByteOrder order) {
        ShortBuffer buffer = ByteBuffer.wrap(bytes).order(order).asShortBuffer();
        short[] shorts = new short[buffer.capacity()];
        buffer.get(shorts);

        // Feed in float values between -1.0f and 1.0f.
        float[] floats = new float[shorts.length];
        for (int i = 0; i < shorts.length; i++) {
            floats[i] = ((float) shorts[i]) / (float) Short.MAX_VALUE;
        }
        return floats;
    }
}

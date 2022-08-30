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

import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ShortBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * {@code FFmpegAudioFactory} is a high performance implementation of {@link AudioFactory} using
 * FFmpeg.
 */
public class FFmpegAudioFactory extends AudioFactory {

    /** {@inheritDoc} */
    @Override
    public Audio fromFile(Path path) throws IOException {
        try (FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(path.toFile())) {
            grabber.start();
            float[] floats = grab(grabber);
            return new Audio(floats, grabber.getSampleRate(), grabber.getAudioChannels());
        } catch (FrameGrabber.Exception e) {
            throw new IOException("Unsupported Audio file", e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public Audio fromInputStream(InputStream is) throws IOException {
        try (FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(is)) {
            grabber.start();
            float[] floats = grab(grabber);
            return new Audio(floats, grabber.getSampleRate(), grabber.getAudioChannels());
        } catch (FrameGrabber.Exception e) {
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

    /**
     * Grab frames from the audio using {@link FFmpegFrameGrabber}.
     *
     * @param grabber the {@link FFmpegFrameGrabber}.
     * @return the float array read from the audio.
     * @throws FFmpegFrameGrabber.Exception if error occurs
     */
    private float[] grab(FFmpegFrameGrabber grabber) throws FFmpegFrameGrabber.Exception {
        List<Float> list = new ArrayList<>();
        Frame frame;
        while ((frame = grabber.grab()) != null) {
            ShortBuffer buffer = (ShortBuffer) frame.samples[0];
            for (int i = 0; i < buffer.limit(); i++) {
                list.add(buffer.get() / (float) Short.MAX_VALUE);
            }
        }
        float[] ret = new float[list.size()];
        for (int i = 0; i < list.size(); i++) {
            ret[i] = list.get(i);
        }
        return ret;
    }
}

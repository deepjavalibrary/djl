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

package ai.djl.audio.dataset;

import ai.djl.audio.featurizer.AudioNormalizer;
import ai.djl.audio.featurizer.AudioProcessor;
import ai.djl.audio.featurizer.LinearSpecgram;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import java.nio.Buffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.bytedeco.javacv.*;

public class AudioData {
    private int sampleRate;
    private int audioChannels;

    private List<AudioProcessor> processorList;

    public AudioData(Configuration configuration) {
        this.sampleRate = configuration.sampleRate;
        this.processorList = configuration.processorList;
    }

    /**
     * Returns a good default {@link AudioData.Configuration} to use for the constructor with
     * defaults.
     *
     * @return a good default {@link AudioData.Configuration} to use for the constructor with
     *     defaults
     */
    public static AudioData.Configuration getDefaultConfiguration() {
        float targetDb = -20f;
        float strideMs = 10f;
        float windowsMs = 20f;
        int sampleRate = 16000;
        List<AudioProcessor> defaultProcessors =
                Arrays.asList(
                        new AudioNormalizer(targetDb),
                        new LinearSpecgram(strideMs, windowsMs, sampleRate));
        return new AudioData.Configuration()
                .setProcessorList(defaultProcessors)
                .setSampleRate(sampleRate);
    }

    private float[] toFloat(String path) {
        List<Float> list = new ArrayList<>();
        float scale = (float) 1.0 / (float) (1 << (8 * 2) - 1);
        System.out.println("test");
        try (FFmpegFrameGrabber audioGrabber = new FFmpegFrameGrabber(path)) {
            audioGrabber.start();
            audioChannels = audioGrabber.getAudioChannels();
            audioGrabber.setSampleRate(sampleRate);
            Frame frame;
            while ((frame = audioGrabber.grabFrame()) != null) {
                Buffer[] buffers = frame.samples;
                //                ShortBuffer[] copiedBuffer = new ShortBuffer[buffers.length];
                //                for (int i = 0; i < buffers.length; i++) {
                //                    deepCopy(buffers[i], copiedBuffer[i]);
                //                }
                ShortBuffer sb = (ShortBuffer) buffers[0];
                for (int i = 0; i < sb.limit(); i++) {
                    list.add(sb.get() * scale);
                }
            }
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }

        float[] floatArray = new float[list.size()];
        int i = 0;
        for (Float f : list) {
            floatArray[i++] = (f != null ? f : Float.NaN);
        }
        return floatArray;
    }

    public NDArray getPreprocessedData(NDManager manager, String path) {
        float[] floatArray = toFloat(path);
        NDArray samples = manager.create(floatArray);
        for (AudioProcessor processor : processorList) {
            samples = processor.extractFeatures(manager, samples);
        }
        return samples;
    }

    private static ShortBuffer deepCopy(ShortBuffer source, ShortBuffer target) {
        int sourceP = source.position();
        int sourceL = source.limit();
        if (null == target) {
            target = ShortBuffer.allocate(source.remaining());
        }
        target.put(source);
        target.flip();
        source.position(sourceP);
        source.limit(sourceL);
        return target;
    }

    public int getAudioChannels() {
        return audioChannels;
    }

    public int getSampleRate() {
        return sampleRate;
    }

    /**
     * The configuration for creating a {@link AudioData} value in a {@link *
     * ai.djl.training.dataset.Dataset}.
     */
    public static final class Configuration {

        private int sampleRate;

        private List<AudioProcessor> processorList;

        public Configuration setProcessorList(List<AudioProcessor> processorList) {
            this.processorList = processorList;
            return this;
        }

        public Configuration setSampleRate(int sampleRate) {
            this.sampleRate = sampleRate;
            return this;
        }

        public AudioData.Configuration update(AudioData.Configuration other) {
            processorList = other.processorList;
            return this;
        }
    }
}

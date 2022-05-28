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

import ai.djl.audio.processor.AudioNormalizer;
import ai.djl.audio.processor.AudioProcessor;
import ai.djl.audio.processor.LinearSpecgram;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import java.nio.Buffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;

/**
 * {@link AudioData} is a utility for managing audio data within a {@link
 * ai.djl.training.dataset.Dataset}. It contains some basic information of an audio file and provide
 * some method to preprocess the original audio file. Since storing all the data into the memory is
 * impossible, this class will only store the path of the original audio data.
 *
 * <p>This class provides a list of {@link AudioProcessor} for user to featurize the data.
 *
 * <p>See {@link SpeechRecognitionDataset} for an example.
 */
public class AudioData {
    private int sampleRate;
    private int audioChannels;

    private List<AudioProcessor> processorList;
    private List<String> audioPaths;

    /**
     * Constructs a new {@link AudioData}.
     *
     * @param configuration the configuration for the {@link AudioData}.
     */
    public AudioData(Configuration configuration) {
        this.sampleRate = configuration.sampleRate;
        this.processorList = configuration.processorList;
    }

    /**
     * Returns a good default {@link AudioData.Configuration} to use for the constructor with
     * defaults.
     *
     * @return a good default {@link AudioData.Configuration} to use for the constructor with
     *     defaults.
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

    /**
     * This method is used for decoding the original audio data and converting it to a float array.
     *
     * @param path The path of the original audio data.
     * @return A float array.
     */
    private float[] toFloat(String path) {
        List<Float> list = new ArrayList<>();
        float scale = (float) 1.0 / (float) (1 << (8 * 2) - 1);
        try (FFmpegFrameGrabber audioGrabber = new FFmpegFrameGrabber(path)) {
            audioGrabber.start();
            audioChannels = audioGrabber.getAudioChannels();
            audioGrabber.setSampleRate(sampleRate);
            Frame frame;
            while ((frame = audioGrabber.grabFrame()) != null) {
                Buffer[] buffers = frame.samples;
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

    /**
     * This method will use a list of {@link AudioProcessor} as featurizer to process the float
     * array.
     *
     * @param manager the manager for converting the float array to {@link NDArray}.
     * @param index The index of path of the original audio data.
     * @return An {@link NDArray} that represent the processed audio data.
     */
    public NDArray getPreprocessedData(NDManager manager, int index) {
        float[] floatArray = toFloat(audioPaths.get(index));
        NDArray samples = manager.create(floatArray);
        for (AudioProcessor processor : processorList) {
            samples = processor.extractFeatures(manager, samples);
        }
        return samples;
    }

    /** @return The number of channels of an audio file. */
    public int getAudioChannels() {
        return audioChannels;
    }

    /** @return The sample rate used by {@link FFmpegFrameGrabber} when sampling the audio file. */
    public int getSampleRate() {
        return sampleRate;
    }

    /** @param audioPaths The path list of original audio data. */
    public void setAudioPaths(List<String> audioPaths) {
        this.audioPaths = audioPaths;
    }

    /** @return The original audio path. */
    public List<String> getAudioPaths() {
        return audioPaths;
    }

    /** @return The total number of audio data in the dataset. */
    public int getTotalSize() {
        return audioPaths.size();
    }

    /**
     * The configuration for creating a {@link AudioData} value in a {@link
     * ai.djl.audio.dataset.AudioData}.
     */
    public static final class Configuration {

        private int sampleRate;

        private List<AudioProcessor> processorList;

        /**
         * @param processorList The list of processor which are used for extracting features from
         *     audio data.
         * @return this configuration.
         */
        public Configuration setProcessorList(List<AudioProcessor> processorList) {
            this.processorList = processorList;
            return this;
        }

        /**
         * @param sampleRate The sampleRate for {@link FFmpegFrameGrabber} to use.
         * @return this configuration.
         */
        public Configuration setSampleRate(int sampleRate) {
            this.sampleRate = sampleRate;
            return this;
        }

        /**
         * Updates this {@link AudioData.Configuration} with the non-null values from another
         * configuration.
         *
         * @param other the other configuration to use to update this
         * @return this configuration after updating
         */
        public AudioData.Configuration update(AudioData.Configuration other) {
            processorList = other.processorList;
            return this;
        }
    }
}

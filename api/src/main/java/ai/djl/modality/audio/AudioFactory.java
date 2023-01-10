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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * {@code AudioFactory} contains audio creation mechanism on top of different platforms like PC and
 * Android. System will choose appropriate Factory based on the supported audio type.
 */
public abstract class AudioFactory {

    private static final Logger logger = LoggerFactory.getLogger(AudioFactory.class);

    private static final String[] FACTORIES = {
        "ai.djl.audio.FFmpegAudioFactory", "ai.djl.modality.audio.SampledAudioFactory"
    };

    protected int channels;
    protected int sampleRate;
    protected int sampleFormat;

    /**
     * Constructs a new instance of {@code AudioFactory}.
     *
     * @return a new instance of {@code AudioFactory}
     */
    public static AudioFactory newInstance() {
        for (String f : FACTORIES) {
            try {
                Class<? extends AudioFactory> clazz =
                        Class.forName(f).asSubclass(AudioFactory.class);
                return clazz.getDeclaredConstructor().newInstance();
            } catch (ReflectiveOperationException e) {
                logger.trace("", e);
            }
        }
        throw new IllegalStateException("Failed to create AudioFactory!");
    }

    /**
     * Returns {@link Audio} from file.
     *
     * @param path the path to the audio
     * @return {@link Audio}
     * @throws IOException Audio not found or not readable
     */
    public abstract Audio fromFile(Path path) throws IOException;

    /**
     * Returns {@link Audio} from URL.
     *
     * @param url the URL to load from
     * @return {@link Audio}
     * @throws IOException URL is not valid.
     */
    public Audio fromUrl(URL url) throws IOException {
        try (InputStream is = url.openStream()) {
            return fromInputStream(is);
        }
    }

    /**
     * Returns {@link Audio} from URL.
     *
     * @param url the String represent URL to load from
     * @return {@link Audio}
     * @throws IOException URL is not valid.
     */
    public Audio fromUrl(String url) throws IOException {
        URI uri = URI.create(url);
        if (uri.isAbsolute()) {
            return fromUrl(uri.toURL());
        }
        return fromFile(Paths.get(url));
    }

    /**
     * Returns {@link Audio} from {@link InputStream}.
     *
     * @param is {@link InputStream}
     * @return {@link Audio}
     * @throws IOException image cannot be read from input stream.
     */
    public abstract Audio fromInputStream(InputStream is) throws IOException;

    /**
     * Returns {@link Audio} from raw data.
     *
     * @param data the raw data in float array form.
     * @return {@link Audio}
     */
    public Audio fromData(float[] data) {
        return new Audio(data);
    }

    /**
     * Returns {@link Audio} from {@link NDArray}.
     *
     * @param array the NDArray with CHW format
     * @return {@link Audio}
     */
    public Audio fromNDArray(NDArray array) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /**
     * Sets the number of channels for {@link AudioFactory} to use.
     *
     * @param channels the number of channels for {@link AudioFactory} to use
     * @return this factory
     */
    public AudioFactory setChannels(int channels) {
        this.channels = channels;
        return this;
    }

    /**
     * Returns the channels of this factory.
     *
     * @return the channels of this factory
     */
    public int getChannels() {
        return channels;
    }

    /**
     * Sets the sampleRate for {@link AudioFactory} to use.
     *
     * @param sampleRate the sampleRate for {@link AudioFactory} to use
     * @return this factory
     */
    public AudioFactory setSampleRate(int sampleRate) {
        this.sampleRate = sampleRate;
        return this;
    }

    /**
     * Returns the sample rate.
     *
     * @return the sample rate in integer
     */
    public int getSampleRate() {
        return sampleRate;
    }

    /**
     * Sets the audio sample format for {@link AudioFactory} to use.
     *
     * @param sampleFormat the sample format
     * @return this factory.
     */
    public AudioFactory setSampleFormat(int sampleFormat) {
        this.sampleFormat = sampleFormat;
        return this;
    }

    /**
     * Returns the sample format name of the audio.
     *
     * @return the format name of the audio
     */
    public int getSampleFormat() {
        return sampleFormat;
    }
}

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

    protected Configuration configuration;

    /**
     * Audio Factory implementation.
     *
     * @param configuration configuration to pass
     */
    public AudioFactory(Configuration configuration) {
        this.configuration = configuration;
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
    public abstract Audio fromData(float[] data);

    /**
     * Returns {@link Audio} from {@link NDArray}.
     *
     * @param array the NDArray with CHW format
     * @return {@link Audio}
     */
    public abstract Audio fromNDArray(NDArray array);

    /** The configuration to set for the factory. */
    public static class Configuration {

        private Integer sampleRate;
        private String format;
        private String audioCodec;

        /**
         * Set the sampleRate for {@link AudioFactory} to use.
         *
         * @param sampleRate The sampleRate for {@link AudioFactory} to use.
         * @return this configuration.
         */
        public Configuration setSampleRate(int sampleRate) {
            this.sampleRate = sampleRate;
            return this;
        }

        /**
         * Get the sample rate.
         *
         * @return Sample rate in integer
         */
        public Integer getSampleRate() {
            return sampleRate;
        }

        /**
         * Set the audio format for {@link AudioFactory} to use.
         *
         * @param format The audio format for {@link AudioFactory} to use.
         * @return this configuration.
         */
        public Configuration setFormat(String format) {
            this.format = format;
            return this;
        }

        /**
         * Get the format name of the audio.
         *
         * @return format
         */
        public String getFormat() {
            return format;
        }

        /**
         * Set the codec for the audio source.
         *
         * @param audioCodec the codec name
         * @return this configuration.
         */
        public Configuration setAudioCodec(String audioCodec) {
            this.audioCodec = audioCodec;
            return this;
        }

        /**
         * Get the codec name of the audio source.
         *
         * @return codec name
         */
        public String getAudioCodec() {
            return this.audioCodec;
        }
    }
}

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

/**
 * {@code Audio} is a container of an audio in DJL. The raw data of the audio is wrapped in a float
 * array.
 */
public class Audio {

    private float[] data;
    private float sampleRate;
    private int channels;

    /**
     * Constructs a new {@code Audio} instance.
     *
     * @param data the wrapped float array data
     */
    public Audio(float[] data) {
        this.data = data;
    }

    /**
     * Constructs a new {@code Audio} instance.
     *
     * @param data the wrapped float array data
     * @param sampleRate the sample rate
     * @param channels number of channels
     */
    public Audio(float[] data, float sampleRate, int channels) {
        this.data = data;
        this.sampleRate = sampleRate;
        this.channels = channels;
    }

    /**
     * Returns the float array data.
     *
     * @return The float array data.
     */
    public float[] getData() {
        return data;
    }

    /**
     * Returns the sample rate.
     *
     * @return sample rate.
     */
    public float getSampleRate() {
        return sampleRate;
    }

    /**
     * Get the number of channels of an audio file.
     *
     * @return The number of channels of an audio file.
     */
    public int getChannels() {
        return channels;
    }
}

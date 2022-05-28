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

import ai.djl.ndarray.NDArray;

/** Useful utils for audio process */
public class AudioUtils {

    /**
     * Calculate root mean square energy in decibels.
     *
     * @param samples input signal, should be 1 dimension array
     * @return root mean square energy
     */
    public static float rmsDb(NDArray samples) {
        samples = samples.pow(2).mean().log10().mul(10);
        return samples.toFloatArray()[0];
    }
}

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

package ai.djl.audio.featurizer;

import ai.djl.audio.AudioUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

public class AudioNormalizer implements AudioProcessor {

    private float target_db;
    static float max_gain_db = 300.0f;

    public AudioNormalizer(float target_db) {
        this.target_db = target_db;
    }

    @Override
    public NDArray extractFeatures(NDManager manager, NDArray samples) {
        float gain = target_db - AudioUtils.rmsDb(samples);
        gain = Math.min(gain, max_gain_db);

        float factor = (float) Math.pow(10f, gain / 20f);
        samples = samples.mul(factor);
        return samples;
    }
}

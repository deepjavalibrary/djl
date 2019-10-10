/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.mxnet.dataset.transform.cv;

import software.amazon.ai.modality.cv.util.NDImageUtils;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.translate.Transform;

/** Normalize an tensor of shape (C, H, W). */
public class Normalize implements Transform {
    private float[] mean;
    private float[] std;

    public Normalize(float[] mean, float[] std) {
        this.mean = mean;
        this.std = std;
    }

    @Override
    public NDArray transform(NDArray array, boolean close) {
        return NDImageUtils.normalize(array, mean, std);
    }
}

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
package org.apache.mxnet.dataset.transform.cv;

import software.amazon.ai.modality.cv.util.NDImageUtils;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.translate.Transform;

/** Resize the image. */
public class Resize implements Transform {
    private int[] size;

    public Resize(int size) {
        this.size = new int[] {size, size};
    }

    public Resize(int[] size) {
        this.size = size;
    }

    @Override
    public NDArray transform(NDArray array, boolean close) {
        return NDImageUtils.resize(array, size);
    }
}

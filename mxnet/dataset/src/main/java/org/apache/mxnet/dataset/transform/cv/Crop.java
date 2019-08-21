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

import org.apache.mxnet.engine.MxImages;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.translate.Transform;

/** Crop the image. The input shape of image is (H, W, C). */
public class Crop implements Transform {
    private int x;
    private int y;
    private int width;
    private int height;

    public Crop(int x, int y, int width, int height) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }

    @Override
    public NDArray transform(NDArray array, boolean close) {
        return MxImages.crop(array, x, y, width, height);
    }
}

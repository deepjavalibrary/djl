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

import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.translate.Transform;

/**
 * Converts an image NDArray to tensor NDArray Converts an image NDArray of shape (H x W x C) in the
 * range [0, 255] to a float32 tensor NDArray of shape (C x H x W) in the range [0, 1].
 */
public class ToTensor implements Transform {

    @Override
    public NDArray transform(NDArray array, boolean close) {
        return NDImageUtils.toTensor(array);
    }
}

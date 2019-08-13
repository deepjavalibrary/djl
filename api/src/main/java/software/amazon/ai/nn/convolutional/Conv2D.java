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
package software.amazon.ai.nn.convolutional;

import software.amazon.ai.Block;
import software.amazon.ai.engine.Engine;

public interface Conv2D extends Convolution {
    /** The Builder to construct a {@link Conv2D} type of {@link Block}. */
    class Builder extends Convolution.Builder<Conv2D> {
        /** {@inheritDoc} */
        @Override
        public Conv2D build() {
            if (kernel == null || numFilters == 0) {
                throw new IllegalArgumentException("Kernel and numFilters must be set");
            }
            return Engine.getInstance()
                    .getNNIndex()
                    .conv2D(kernel, stride, pad, dilate, numFilters, numGroups, includeBias);
        }
    }
}

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
package software.amazon.ai.nn.norm;

import software.amazon.ai.Block;
import software.amazon.ai.engine.Engine;
import software.amazon.ai.ndarray.NDArray;

public interface BatchNorm extends Block {

    NDArray forward(NDArray data);

    public static final class Builder {

        private int axis = 1;
        private float epsilon = 1E-5f;
        private float momentum = .9f;

        public Builder setAxis(int val) {
            axis = val;
            return this;
        }

        public Builder setEpsilon(float val) {
            epsilon = val;
            return this;
        }

        public Builder setMomentum(float val) {
            momentum = val;
            return this;
        }

        public BatchNorm build() {
            return Engine.getInstance().getNNIndex().batchNorm2D(axis, epsilon, momentum);
        }
    }
}

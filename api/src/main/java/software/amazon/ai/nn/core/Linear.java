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
package software.amazon.ai.nn.core;

import software.amazon.ai.Block;
import software.amazon.ai.engine.Engine;
import software.amazon.ai.ndarray.NDArray;

public interface Linear extends Block {

    NDArray forward(NDArray data);

    class Builder {

        private int units;

        public int getUnits() {
            return units;
        }

        public Builder setUnits(int units) {
            this.units = units;
            return this;
        }

        public Linear build() {
            if (units == 0) {
                throw new IllegalStateException("You must specify units");
            }
            return Engine.getInstance().getNNIndex().linear(units);
        }
    }
}

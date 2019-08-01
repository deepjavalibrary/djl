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
import software.amazon.ai.training.Activation;

public interface Prelu extends Activation {

    /** {@inheritDoc} */
    @Override
    NDArray forward(NDArray data);

    /** The Builder to construct a {@link Prelu} type of {@link Block}. */
    class Builder {

        /**
         * Returns the constructed {@code Prelu}.
         *
         * @return Returns the constructed {@code Prelu}
         * @throws IllegalArgumentException Thrown if all required parameters (outChannels) have not
         *     been set
         */
        public Prelu build() {
            return Engine.getInstance().getNNIndex().prelu();
        }
    }
}

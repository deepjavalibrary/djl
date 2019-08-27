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
package software.amazon.ai.nn.recurrent;

import software.amazon.ai.engine.Engine;
import software.amazon.ai.nn.Block;

public interface RNN extends RecurrentCell {

    /** The Builder to construct a {@link RNN} type of {@link Block}. */
    final class Builder extends RecurrentCell.Builder<RNN> {
        /** {@inheritDoc} */
        @Override
        public RNN build() {
            if (stateSize == -1 || numStackedLayers == -1) {
                throw new IllegalArgumentException("Must set stateSize and numStackedLayers");
            }
            return Engine.getInstance().getNNIndex().rnn(this);
        }
    }

    enum Activation {
        RELU,
        TANH
    }
}

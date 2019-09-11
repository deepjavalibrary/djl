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

package software.amazon.ai.training.optimizer;

import software.amazon.ai.training.optimizer.learningrate.LrTracker;

/** An NAG optimizer. Build with {@link Nag.Builder}. */
public interface Nag extends Optimizer {

    class Builder extends BaseBuilder<Builder> {

        private LrTracker lrTracker;
        private float momentum;

        public Builder setLrTracker(LrTracker lrTracker) {
            this.lrTracker = lrTracker;
            return this;
        }

        public Builder setMomentum(float momentum) {
            this.momentum = momentum;
            return this;
        }

        public LrTracker getLrTracker() {
            return lrTracker;
        }

        public float getMomentum() {
            return momentum;
        }

        @Override
        protected Builder self() {
            return this;
        }

        public Nag build() {
            if (lrTracker == null) {
                throw new IllegalArgumentException("No lrTracker set");
            }
            if (momentum == 0) {
                throw new IllegalArgumentException("The momentum should be set");
            }
            return factory.createNag(this);
        }
    }
}

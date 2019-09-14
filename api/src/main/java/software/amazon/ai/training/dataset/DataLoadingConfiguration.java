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
package software.amazon.ai.training.dataset;

import java.util.concurrent.ExecutorService;
import software.amazon.ai.Device;

/**
 * DataLoadingConfiguration is used to build data loading configuration. It allows users to
 * customize loading order, automatic batching and optimize performance with multi-thread and memory
 * pining.
 */
public final class DataLoadingConfiguration {
    private ExecutorService executor;
    private Device pinDevice;

    private DataLoadingConfiguration(Builder builder) {
        this.executor = builder.executor;
        this.pinDevice = builder.pinDevice;
    }

    public ExecutorService getExecutor() {
        return executor;
    }

    public Device getPinDevice() {
        return pinDevice;
    }

    public static final class Builder {
        private ExecutorService executor;

        private Device pinDevice;

        public Builder setExcutor(ExecutorService executor) {
            this.executor = executor;
            return this;
        }

        public Builder setPinDevice(Device ctx) {
            this.pinDevice = ctx;
            return this;
        }

        public DataLoadingConfiguration build() {
            // sampler is exclusive with shuffle ()
            return new DataLoadingConfiguration(this);
        }
    }
}

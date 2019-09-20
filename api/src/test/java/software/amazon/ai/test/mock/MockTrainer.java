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
package software.amazon.ai.test.mock;

import software.amazon.ai.metric.Metrics;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.training.ModelSaver;
import software.amazon.ai.training.Trainer;

public class MockTrainer implements Trainer {

    @Override
    public void step() {}

    @Override
    public NDList forward(NDList intput) {
        return null;
    }

    @Override
    public void setMetrics(Metrics metrics) {}

    @Override
    public NDManager getManager() {
        return null;
    }

    @Override
    public ModelSaver getModelSaver() {
        return null;
    }

    @Override
    public void setModelSaver(ModelSaver modelSaver) {}

    @Override
    public void checkpoint() {}

    @Override
    public void close() {}
}

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

package software.amazon.ai.integration.tests;

import org.apache.mxnet.engine.MxParameterStore;
import org.testng.annotations.Test;
import software.amazon.ai.Model;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.training.ParameterStore;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.training.optimizer.Sgd;
import software.amazon.ai.training.optimizer.learningrate.LearningRateTracker;

public class ParameterStoreTest {

    @Test
    public void testParameterStore() throws FailedTestException {
        try (Model model = Model.newInstance()) {
            NDManager manager = model.getNDManager();
            NDArray weight = manager.create(new float[] {1.f, 1.f}, new Shape(1, 2));
            NDArray grad = manager.create(new float[] {2.f, 2.f}, new Shape(1, 2));

            Optimizer optimizer =
                    new Sgd.Builder()
                            .setRescaleGrad(1.0f / 32)
                            .setLearningRateTracker(LearningRateTracker.fixedLR(.03f))
                            .build();
            try (ParameterStore ps = new MxParameterStore(false, optimizer)) {
                ps.init(0, weight);
                ps.push(0, grad);
                ps.pull(0, weight);
                Assertions.assertEquals(
                        weight,
                        grad,
                        "Parameter Store updated wrong result: actual "
                                + weight
                                + ", expected "
                                + grad);
            }
        }
    }
}

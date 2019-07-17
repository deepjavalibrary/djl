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

import org.apache.mxnet.engine.MxAutograd;
import org.apache.mxnet.engine.MxNDArray;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.AbstractTest;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;

public class MxAutoGradIntegrationTest extends AbstractTest {
    // TODO use API level integration test once moved Autograd to API package

    public static void main(String[] args) {
        new MxAutoGradIntegrationTest().runTest(args);
    }

    @RunAsTest
    public void testAutograd() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager();
                MxAutograd autograd = new MxAutograd()) {

            NDArray lhs = manager.create(new float[] {6, -9, -12, 15, 0, 4}, new Shape(2, 3));
            NDArray rhs = manager.create(new float[] {2, 3, -4}, new Shape(3, 1));
            autograd.attachGradient(lhs);
            // autograd automatically set recording and training during initialization
            Assertions.assertTrue(autograd.isRecording());
            Assertions.assertTrue(autograd.isTraining());
            NDArray result = NDArrays.mmul(lhs, rhs);
            autograd.backward((MxNDArray) result);
        }
    }
}

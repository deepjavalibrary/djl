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
package software.amazon.ai.training;

import software.amazon.ai.ndarray.NDArray;

public final class Loss {

    private Loss() {}

    public static NDArray l2Loss(NDArray pred, NDArray label, float weight, int batchAxis) {
        return pred.getNDArrayInternal().l2Loss(label, weight, batchAxis);
    }

    public static NDArray softmaxCrossEntropyLoss(
            NDArray pred,
            NDArray label,
            float weight,
            int batchAxis,
            int axis,
            boolean sparseLabel,
            boolean fromLogit) {
        return pred.getNDArrayInternal()
                .softmaxCrossEntropyLoss(label, weight, batchAxis, axis, sparseLabel, fromLogit);
    }
}

/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.training.loss;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

/**
 * Calculates the loss for tabNet. Actually, loss has been calculated through the forward function
 * of tabNet. What's done here is just getting the loss function from prediction.
 */
public class TabNetLoss extends Loss {
    /** Calculates the loss of a TabNet instance. */
    public TabNetLoss() {
        this("TabNetLoss");
    }

    /**
     * Calculates the loss of a TabNet instance.
     *
     * @param name the name of the loss function
     */
    public TabNetLoss(String name) {
        super(name);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        // loss is already calculated inside the forward of tabNet
        // so here we just need to get it out from prediction
        return predictions.get(1);
    }
}

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
 * Calculates the loss for tabNet. I set the loss as the combination of L2Loss and sparseLoss
 * together for airfoil dateset. If the output of tabNet should be classification and so on, the
 * loss should set as softmaxLoss. Actually, tabNet is not only used for Supervised Learning, it's
 * also widely used in unsupervised learning. For unsupervised learning, it should come from the
 * decoder(aka attentionTransformer of tabNet)
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
        // sparseLoss is already calculated inside the forward of tabNet
        // so here we just need to get it out from prediction
        return labels.singletonOrThrow()
                .sub(predictions.get(0))
                .square()
                .mean()
                .add(predictions.get(1));
    }
}

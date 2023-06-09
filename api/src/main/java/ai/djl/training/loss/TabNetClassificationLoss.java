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
 * Calculates the loss for tabNet in Classification tasks.
 *
 * <p>Actually, tabNet is not only used for Supervised Learning, it's also widely used in
 * unsupervised learning. For unsupervised learning, it should come from the decoder(aka
 * attentionTransformer of tabNet)
 */
public final class TabNetClassificationLoss extends Loss {
    /** Calculates the loss of a TabNet instance for regression tasks. */
    public TabNetClassificationLoss() {
        this("TabNetClassificationLoss");
    }

    /**
     * Calculates the loss of a TabNet instance for regression tasks.
     *
     * @param name the name of the loss function
     */
    public TabNetClassificationLoss(String name) {
        super(name);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        return Loss.softmaxCrossEntropyLoss()
                .evaluate(labels, new NDList(predictions.get(0)))
                .add(predictions.get(1).mean());
    }
}

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
package ai.djl.training.listener;

import ai.djl.training.Trainer;

/**
 * Base implementation of the training listener that does nothing. This is to be used as a base
 * class for custom training listeners that just want to listen to one event, so it is not necessary
 * to override methods you do not care for.
 */
public abstract class TrainingListenerAdapter implements TrainingListener {

    /** {@inheritDoc} */
    @Override
    public void onEpoch(Trainer trainer) {}

    /** {@inheritDoc} */
    @Override
    public void onTrainingBatch(Trainer trainer, BatchData batchData) {}

    /** {@inheritDoc} */
    @Override
    public void onValidationBatch(Trainer trainer, BatchData batchData) {}

    /** {@inheritDoc} */
    @Override
    public void onTrainingBegin(Trainer trainer) {}

    /** {@inheritDoc} */
    @Override
    public void onTrainingEnd(Trainer trainer) {}
}

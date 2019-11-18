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
package ai.djl.training;

/**
 * {@code TrainingListener} offers an interface that allows performing some actions when certain
 * events have occurred in the {@link Trainer}.
 *
 * <p>The methods {@link #onEpoch() onEpoch}, {@link #onTrainingBatch() onTrainingBatch}, {@link
 * #onValidationBatch() onValidationBatch} are called during training. Adding an implementation of
 * the listener to the {@link Trainer} allows performing any desired actions at those junctures.
 * These could be used for collection metrics, or logging, or any other purpose.
 */
public interface TrainingListener {

    /** Listens to the end of an epoch during training. */
    void onEpoch();

    /** Listens to the end of training one batch of data during training. */
    void onTrainingBatch();

    /** Listens to the end of validating one batch of data during validation. */
    void onValidationBatch();
}

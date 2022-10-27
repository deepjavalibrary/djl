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
package ai.djl.training.tracker;

/**
 * A {@code Tracker} represents a collection of hyperparameters or {@link Tracker}s that changes
 * gradually through the training process.
 *
 * @see Tracker
 */
public interface ParameterTracker {

    /**
     * Fetches the value after the given number of steps/updates for the parameter.
     *
     * @param parameterId the id of the parameter to get the new value for
     * @param numUpdate the total number of steps/updates
     * @return this {@code Builder}
     */
    float getNewValue(String parameterId, int numUpdate);
}

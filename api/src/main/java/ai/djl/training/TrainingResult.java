/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import java.util.Collections;
import java.util.Map;

/** A class that is responsible for holding the training result produced by {@link Trainer}. */
public class TrainingResult {

    private int epoch;
    private Map<String, Float> evaluations = Collections.emptyMap();

    /**
     * Returns the train loss.
     *
     * @return the train loss
     */
    public Float getTrainLoss() {
        return evaluations.get("train_loss");
    }

    /**
     * Returns the validate loss.
     *
     * @return the validate loss
     */
    public Float getValidateLoss() {
        return evaluations.get("validate_loss");
    }

    /**
     * Returns the evaluation to which the specified key is mapped.
     *
     * @param key the key whose associated value is to be returned
     * @return the evaluation to which the specified key is mapped
     */
    public Float getTrainEvaluation(String key) {
        return evaluations.get("train_" + key);
    }

    /**
     * Returns the evaluation to which the specified key is mapped.
     *
     * @param key the key whose associated value is to be returned
     * @return the evaluation to which the specified key is mapped
     */
    public Float getValidateEvaluation(String key) {
        return evaluations.get("validate_" + key);
    }

    /**
     * Returns the actual number of epoch.
     *
     * @return the actual number of epoch
     */
    public int getEpoch() {
        return epoch;
    }

    /**
     * Sets the actual number of epoch.
     *
     * @param epoch the actual number of epoch
     */
    public void setEpoch(int epoch) {
        this.epoch = epoch;
    }

    /**
     * Returns the raw evaluation metrics.
     *
     * @return the raw evaluation metrics
     */
    public Map<String, Float> getEvaluations() {
        return evaluations;
    }

    /**
     * Sets the raw evaluation metrics.
     *
     * @param evaluations the raw evaluation metrics
     */
    public void setEvaluations(Map<String, Float> evaluations) {
        this.evaluations = evaluations;
    }
}

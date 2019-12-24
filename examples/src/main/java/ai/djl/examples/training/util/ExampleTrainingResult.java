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
package ai.djl.examples.training.util;

public class ExampleTrainingResult {

    public static final ExampleTrainingResult FAILURE =
            new ExampleTrainingResult().setSuccess(false);

    protected boolean success;
    protected float trainingAccuracy;
    protected float trainingLoss;
    protected float validationAccuracy;
    protected float validationLoss;

    public boolean isSuccess() {
        return success;
    }

    public ExampleTrainingResult setSuccess(boolean success) {
        this.success = success;
        return this;
    }

    public float getTrainingAccuracy() {
        return trainingAccuracy;
    }

    public ExampleTrainingResult setTrainingAccuracy(float trainingAccuracy) {
        this.trainingAccuracy = trainingAccuracy;
        return this;
    }

    public float getTrainingLoss() {
        return trainingLoss;
    }

    public ExampleTrainingResult setTrainingLoss(float trainingLoss) {
        this.trainingLoss = trainingLoss;
        return this;
    }

    public float getValidationAccuracy() {
        return validationAccuracy;
    }

    public ExampleTrainingResult setValidationAccuracy(float validationAccuracy) {
        this.validationAccuracy = validationAccuracy;
        return this;
    }

    public float getValidationLoss() {
        return validationLoss;
    }

    public ExampleTrainingResult setValidationLoss(float validationLoss) {
        this.validationLoss = validationLoss;
        return this;
    }
}

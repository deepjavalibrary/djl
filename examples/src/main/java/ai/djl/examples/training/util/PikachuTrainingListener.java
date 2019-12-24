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

import ai.djl.metric.Metric;
import ai.djl.metric.Metrics;
import ai.djl.training.Trainer;
import ai.djl.training.loss.Loss;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PikachuTrainingListener extends ExampleTrainingListener {

    private static final Logger logger = LoggerFactory.getLogger(ExampleTrainingListener.class);

    private float trainingClassAccuracy;
    private float trainingBoundingBoxError;
    private float validationClassAccuracy;
    private float validationBoundingBoxError;

    public PikachuTrainingListener(int batchSize, int trainDataSize, int validateDataSize) {
        super(batchSize, trainDataSize, validateDataSize);
    }

    @Override
    public String getTrainingStatus(Trainer trainer) {
        Metrics metrics = trainer.getMetrics();
        Loss loss = trainer.getLoss();
        StringBuilder sb = new StringBuilder();
        List<Metric> list = metrics.getMetric("train_" + loss.getName());
        trainingLoss = list.get(list.size() - 1).getValue().floatValue();

        list = metrics.getMetric("train_classAccuracy");
        trainingClassAccuracy = list.get(list.size() - 1).getValue().floatValue();

        list = metrics.getMetric("train_boundingBoxError");
        trainingBoundingBoxError = list.get(list.size() - 1).getValue().floatValue();
        sb.append(
                String.format(
                        "loss: %2.3ef, classAccuracy: %.4f, bboxError: %2.3e,",
                        trainingLoss, trainingClassAccuracy, trainingBoundingBoxError));

        list = metrics.getMetric("train");
        if (!list.isEmpty()) {
            float batchTime = list.get(list.size() - 1).getValue().longValue() / 1_000_000_000f;
            sb.append(String.format(" speed: %.2f images/sec", (float) batchSize / batchTime));
        }
        return sb.toString();
    }

    @Override
    public void printTrainingStatus(Trainer trainer) {
        Metrics metrics = trainer.getMetrics();
        Loss loss = trainer.getLoss();
        List<Metric> list = metrics.getMetric("train_" + loss.getName());
        trainingLoss = list.get(list.size() - 1).getValue().floatValue();

        list = metrics.getMetric("train_classAccuracy");
        trainingClassAccuracy = list.get(list.size() - 1).getValue().floatValue();

        list = metrics.getMetric("train_boundingBoxError");
        trainingBoundingBoxError = list.get(list.size() - 1).getValue().floatValue();

        logger.info(
                "train loss: {}, train class accuracy: {}, train bounding box error: {}",
                trainingLoss,
                trainingClassAccuracy,
                trainingBoundingBoxError);
        list = metrics.getMetric("validate_" + loss.getName());
        if (!list.isEmpty()) {
            validationLoss = list.get(list.size() - 1).getValue().floatValue();
            list = metrics.getMetric("validate_classAccuracy");
            validationClassAccuracy = list.get(list.size() - 1).getValue().floatValue();
            list = metrics.getMetric("validate_boundingBoxError");
            validationBoundingBoxError = list.get(list.size() - 1).getValue().floatValue();
            logger.info(
                    "validate loss: {}, validate class accuracy: {}, validate bounding box error: {}",
                    validationLoss,
                    validationClassAccuracy,
                    validationBoundingBoxError);
        } else {
            logger.info("validation has not been run.");
        }
    }

    @Override
    public PikachuTrainingResult getResult() {
        PikachuTrainingResult result = new PikachuTrainingResult();
        result.setSuccess(true)
                .setTrainingAccuracy(trainingAccuracy)
                .setTrainingLoss(trainingLoss)
                .setValidationAccuracy(validationAccuracy)
                .setValidationLoss(validationLoss);
        return result.setValidationClassAccuracy(validationClassAccuracy)
                .setValidationBoundingBoxError(validationBoundingBoxError);
    }
}

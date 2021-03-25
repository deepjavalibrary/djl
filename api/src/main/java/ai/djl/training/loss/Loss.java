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
package ai.djl.training.loss;

import ai.djl.ndarray.NDList;
import ai.djl.training.evaluator.Evaluator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Loss functions (or Cost functions) are used to evaluate the model predictions against true labels
 * for optimization.
 *
 * <p>Although all evaluators can be used to measure the performance of a model, not all of them are
 * suited to being used by an optimizer. Loss functions are usually non-negative where a larger loss
 * represents worse performance. They are also real-valued to accurately compare models.
 *
 * <p>When creating a loss function, you should avoid having the loss depend on the batch size. For
 * example, if you have a loss per item in a batch and sum those losses, your loss would be {@code
 * numItemsInBatch*avgLoss}. Instead, you should take the mean of those losses to reduce out the
 * batchSize factor. Otherwise, it can make it difficult to tune the learning rate since any change
 * in the batch size would throw it off. If you have a variable batch size, it would be even more
 * difficult.
 *
 * <p>For more details about the class internals, see {@link Evaluator}.
 */
public abstract class Loss extends Evaluator {

    private Map<String, Float> totalLoss;

    /**
     * Base class for metric with abstract update methods.
     *
     * @param name The display name of the Loss
     */
    public Loss(String name) {
        super(name);
        totalLoss = new ConcurrentHashMap<>();
    }

    /**
     * Returns a new instance of {@link L1Loss} with default weight and batch axis.
     *
     * @return a new instance of {@link L1Loss}
     */
    public static L1Loss l1Loss() {
        return new L1Loss();
    }

    /**
     * Returns a new instance of {@link L1Loss} with default weight and batch axis.
     *
     * @param name the name of the loss
     * @return a new instance of {@link L1Loss}
     */
    public static L1Loss l1Loss(String name) {
        return new L1Loss(name);
    }

    /**
     * Returns a new instance of {@link L1Loss} with given weight and batch axis.
     *
     * @param name the name of the loss
     * @param weight the weight to apply on loss value, default 1
     * @return a new instance of {@link L1Loss}
     */
    public static L1Loss l1Loss(String name, float weight) {
        return new L1Loss(name, weight);
    }

    /**
     * Returns a new instance of {@link L2Loss} with default weight and batch axis.
     *
     * @return a new instance of {@link L2Loss}
     */
    public static L2Loss l2Loss() {
        return new L2Loss();
    }

    /**
     * Returns a new instance of {@link L2Loss} with default weight and batch axis.
     *
     * @param name the name of the loss
     * @return a new instance of {@link L2Loss}
     */
    public static L2Loss l2Loss(String name) {
        return new L2Loss(name);
    }

    /**
     * Returns a new instance of {@link L2Loss} with given weight and batch axis.
     *
     * @param name the name of the loss
     * @param weight the weight to apply on loss value, default 1
     * @return a new instance of {@link L2Loss}
     */
    public static L2Loss l2Loss(String name, float weight) {
        return new L2Loss(name, weight);
    }

    /**
     * Returns a new instance of {@link SigmoidBinaryCrossEntropyLoss} with default arguments.
     *
     * @return a new instance of {@link SigmoidBinaryCrossEntropyLoss}
     */
    public static SigmoidBinaryCrossEntropyLoss sigmoidBinaryCrossEntropyLoss() {
        return new SigmoidBinaryCrossEntropyLoss();
    }

    /**
     * Returns a new instance of {@link SigmoidBinaryCrossEntropyLoss} with default arguments.
     *
     * @param name the name of the loss
     * @return a new instance of {@link SigmoidBinaryCrossEntropyLoss}
     */
    public static SigmoidBinaryCrossEntropyLoss sigmoidBinaryCrossEntropyLoss(String name) {
        return new SigmoidBinaryCrossEntropyLoss(name);
    }

    /**
     * Returns a new instance of {@link SigmoidBinaryCrossEntropyLoss} with the given arguments.
     *
     * @param name the name of the loss
     * @param weight the weight to apply on the loss value, default 1
     * @param fromSigmoid whether the input is from the output of sigmoid, default false
     * @return a new instance of {@link SigmoidBinaryCrossEntropyLoss}
     */
    public static SigmoidBinaryCrossEntropyLoss sigmoidBinaryCrossEntropyLoss(
            String name, float weight, boolean fromSigmoid) {
        return new SigmoidBinaryCrossEntropyLoss(name, weight, fromSigmoid);
    }

    /**
     * Returns a new instance of {@link SoftmaxCrossEntropyLoss} with default arguments.
     *
     * @return a new instance of {@link SoftmaxCrossEntropyLoss}
     */
    public static SoftmaxCrossEntropyLoss softmaxCrossEntropyLoss() {
        return new SoftmaxCrossEntropyLoss();
    }

    /**
     * Returns a new instance of {@link SoftmaxCrossEntropyLoss} with default arguments.
     *
     * @param name the name of the loss
     * @return a new instance of {@link SoftmaxCrossEntropyLoss}
     */
    public static SoftmaxCrossEntropyLoss softmaxCrossEntropyLoss(String name) {
        return new SoftmaxCrossEntropyLoss(name);
    }

    /**
     * Returns a new instance of {@link SoftmaxCrossEntropyLoss} with the given arguments.
     *
     * @param name the name of the loss
     * @param weight the weight to apply on the loss value, default 1
     * @param classAxis the axis that represents the class probabilities, default -1
     * @param sparseLabel whether labels are integer array or probabilities, default true
     * @param fromLogit whether labels are log probabilities or un-normalized numbers
     * @return a new instance of {@link SoftmaxCrossEntropyLoss}
     */
    public static SoftmaxCrossEntropyLoss softmaxCrossEntropyLoss(
            String name, float weight, int classAxis, boolean sparseLabel, boolean fromLogit) {
        return new SoftmaxCrossEntropyLoss(name, weight, classAxis, sparseLabel, fromLogit);
    }

    /**
     * Returns a new instance of {@link MaskedSoftmaxCrossEntropyLoss} with default arguments.
     *
     * @return a new instance of {@link MaskedSoftmaxCrossEntropyLoss}
     */
    public static MaskedSoftmaxCrossEntropyLoss maskedSoftmaxCrossEntropyLoss() {
        return new MaskedSoftmaxCrossEntropyLoss();
    }

    /**
     * Returns a new instance of {@link MaskedSoftmaxCrossEntropyLoss} with default arguments.
     *
     * @param name the name of the loss
     * @return a new instance of {@link MaskedSoftmaxCrossEntropyLoss}
     */
    public static MaskedSoftmaxCrossEntropyLoss maskedSoftmaxCrossEntropyLoss(String name) {
        return new MaskedSoftmaxCrossEntropyLoss(name);
    }

    /**
     * Returns a new instance of {@link MaskedSoftmaxCrossEntropyLoss} with the given arguments.
     *
     * @param name the name of the loss
     * @param weight the weight to apply on the loss value, default 1
     * @param classAxis the axis that represents the class probabilities, default -1
     * @param sparseLabel whether labels are integer array or probabilities, default true
     * @param fromLogit whether labels are log probabilities or un-normalized numbers
     * @return a new instance of {@link MaskedSoftmaxCrossEntropyLoss}
     */
    public static MaskedSoftmaxCrossEntropyLoss maskedSoftmaxCrossEntropyLoss(
            String name, float weight, int classAxis, boolean sparseLabel, boolean fromLogit) {
        return new MaskedSoftmaxCrossEntropyLoss(name, weight, classAxis, sparseLabel, fromLogit);
    }

    /**
     * Returns a new instance of {@link HingeLoss} with default arguments.
     *
     * @return a new instance of {@link HingeLoss}
     */
    public static HingeLoss hingeLoss() {
        return new HingeLoss();
    }

    /**
     * Returns a new instance of {@link HingeLoss} with default arguments.
     *
     * @param name the name of the loss
     * @return a new instance of {@link HingeLoss}
     */
    public static HingeLoss hingeLoss(String name) {
        return new HingeLoss(name);
    }

    /**
     * Returns a new instance of {@link HingeLoss} with the given arguments.
     *
     * @param name the name of the loss
     * @param margin the margin in hinge loss. Defaults to 1.0
     * @param weight the weight to apply on loss value, default 1
     * @return a new instance of {@link HingeLoss}
     */
    public static HingeLoss hingeLoss(String name, int margin, float weight) {
        return new HingeLoss(name, margin, weight);
    }

    /**
     * Returns a new instance of {@link L1WeightDecay} with default weight and name.
     *
     * @param parameters holds the model weights that will be penalized
     * @return a new instance of {@link L1WeightDecay}
     */
    public static L1WeightDecay l1WeightedDecay(NDList parameters) {
        return new L1WeightDecay(parameters);
    }

    /**
     * Returns a new instance of {@link L1WeightDecay} with default weight.
     *
     * @param name the name of the weight decay
     * @param parameters holds the model weights that will be penalized
     * @return a new instance of {@link L1WeightDecay}
     */
    public static L1WeightDecay l1WeightedDecay(String name, NDList parameters) {
        return new L1WeightDecay(name, parameters);
    }

    /**
     * Returns a new instance of {@link L1WeightDecay}.
     *
     * @param name the name of the weight decay
     * @param weight the weight to apply on weight decay value, default 1
     * @param parameters holds the model weights that will be penalized
     * @return a new instance of {@link L1WeightDecay}
     */
    public static L1WeightDecay l1WeightedDecay(String name, float weight, NDList parameters) {
        return new L1WeightDecay(name, parameters, weight);
    }

    /**
     * Returns a new instance of {@link L2WeightDecay} with default weight and name.
     *
     * @param parameters holds the model weights that will be penalized
     * @return a new instance of {@link L2WeightDecay}
     */
    public static L2WeightDecay l2WeightedDecay(NDList parameters) {
        return new L2WeightDecay(parameters);
    }

    /**
     * Returns a new instance of {@link L2WeightDecay} with default weight.
     *
     * @param name the name of the weight decay
     * @param parameters holds the model weights that will be penalized
     * @return a new instance of {@link L2WeightDecay}
     */
    public static L2WeightDecay l2WeightedDecay(String name, NDList parameters) {
        return new L2WeightDecay(name, parameters);
    }

    /**
     * Returns a new instance of {@link L2WeightDecay}.
     *
     * @param name the name of the weight decay
     * @param weight the weight to apply on weight decay value, default 1
     * @param parameters holds the model weights that will be penalized
     * @return a new instance of {@link L2WeightDecay}
     */
    public static L2WeightDecay l2WeightedDecay(String name, float weight, NDList parameters) {
        return new L2WeightDecay(name, parameters, weight);
    }

    /**
     * Returns a new instance of {@link ElasticNetWeightDecay} with default weight and name.
     *
     * @param parameters holds the model weights that will be penalized
     * @return a new instance of {@link ElasticNetWeightDecay}
     */
    public static ElasticNetWeightDecay elasticNetWeightedDecay(NDList parameters) {
        return new ElasticNetWeightDecay(parameters);
    }

    /**
     * Returns a new instance of {@link ElasticNetWeightDecay} with default weight.
     *
     * @param name the name of the weight decay
     * @param parameters holds the model weights that will be penalized
     * @return a new instance of {@link ElasticNetWeightDecay}
     */
    public static ElasticNetWeightDecay elasticNetWeightedDecay(String name, NDList parameters) {
        return new ElasticNetWeightDecay(name, parameters);
    }

    /**
     * Returns a new instance of {@link ElasticNetWeightDecay}.
     *
     * @param name the name of the weight decay
     * @param weight the weight to apply on weight decay values, default 1
     * @param parameters holds the model weights that will be penalized
     * @return a new instance of {@link ElasticNetWeightDecay}
     */
    public static ElasticNetWeightDecay elasticNetWeightedDecay(
            String name, float weight, NDList parameters) {
        return new ElasticNetWeightDecay(name, parameters, weight);
    }

    /**
     * Returns a new instance of {@link ElasticNetWeightDecay}.
     *
     * @param name the name of the weight decay
     * @param weight1 the weight to apply on weight decay L1 value, default 1
     * @param weight2 the weight to apply on weight decay L2 value, default 1
     * @param parameters holds the model weights that will be penalized
     * @return a new instance of {@link ElasticNetWeightDecay}
     */
    public static ElasticNetWeightDecay elasticNetWeightedDecay(
            String name, float weight1, float weight2, NDList parameters) {
        return new ElasticNetWeightDecay(name, parameters, weight1, weight2);
    }

    /** {@inheritDoc} */
    @Override
    public void addAccumulator(String key) {
        totalInstances.put(key, 0L);
        totalLoss.put(key, 0f);
    }

    /** {@inheritDoc} */
    @Override
    public void updateAccumulator(String key, NDList labels, NDList predictions) {
        // this is a synchronized operation, only call it at end of batch or epoch
        float update = evaluate(labels, predictions).sum().getFloat();
        totalInstances.compute(key, (k, v) -> v + 1);
        totalLoss.compute(key, (k, v) -> v + update);
    }

    /** {@inheritDoc} */
    @Override
    public void resetAccumulator(String key) {
        totalInstances.compute(key, (k, v) -> 0L);
        totalLoss.compute(key, (k, v) -> 0f);
    }

    /** {@inheritDoc} */
    @Override
    public float getAccumulator(String key) {
        Long total = totalInstances.get(key);
        if (total == null) {
            throw new IllegalArgumentException("No loss found at that path");
        }

        if (total == 0) {
            return Float.NaN;
        }

        return totalLoss.get(key) / totalInstances.get(key);
    }
}

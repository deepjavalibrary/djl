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

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.training.loss.Loss;
import java.util.List;

/**
 * An interface that represents a single training iteration.
 *
 * <p>{@code Trainer} provides an easy, and manageable interface for training. {@code Trainer} is
 * not thread-safe.
 *
 * <p>See the tutorials on:
 *
 * <ul>
 *   <li><a
 *       href="https://github.com/awslabs/djl/blob/master/jupyter/train_your_first_model.ipynb">Training
 *       your first model</a>
 *   <li><a
 *       href="https://github.com/awslabs/djl/blob/master/jupyter/transfer_learning_on_cifar10.ipynb">Training
 *       using transfer learning</a>
 *   <li><a
 *       href="https://github.com/awslabs/djl/blob/master/jupyter/load_mxnet_model.ipynb">Inference
 *       with an MXNet model</a>
 * </ul>
 */
public interface Trainer extends AutoCloseable {

    /**
     * Initializes the {@link Model} that the {@code Trainer} is going to train.
     *
     * @param shapes an array of {@code Shape} of the inputs
     */
    void initialize(Shape... shapes);

    /**
     * Fetches an iterator that can iterate through the given {@link Dataset}.
     *
     * @param dataset the dataset to iterate through
     * @return an {@link Iterable} of {@link Batch} that contains batches of data from the dataset
     */
    default Iterable<Batch> iterateDataset(Dataset dataset) {
        return dataset.getData(getManager());
    }

    /**
     * Returns a new instance of {@link GradientCollector}.
     *
     * @return a new instance of {@link GradientCollector}
     */
    GradientCollector newGradientCollector();

    /**
     * Trains the model with one iteration of the given {@link Batch} of data.
     *
     * @param batch a {@link Batch} that contains data, and its respective labels
     */
    void trainBatch(Batch batch);

    /**
     * Applies the forward function of the model once on the given input {@link NDList}.
     *
     * @param input the input {@link NDList}
     * @return the output of the forward function
     */
    NDList forward(NDList input);

    /**
     * Validates the given batch of data.
     *
     * <p>During validation, the evaluators and losses are computed, but gradients aren't computed,
     * and parameters aren't updated.
     *
     * @param batch a {@link Batch} of data
     */
    void validateBatch(Batch batch);

    /** Updates all of the parameters of the model once. */
    void step();

    /**
     * Returns the Metrics param used for benchmarking.
     *
     * @return the the Metrics param used for benchmarking.
     */
    Metrics getMetrics();

    /**
     * Attaches a Metrics param to use for benchmarking.
     *
     * @param metrics the Metrics class
     */
    void setMetrics(Metrics metrics);

    /**
     * Returns the devices used for training.
     *
     * @return the devices used for training
     */
    List<Device> getDevices();

    /** Resets each of the evaluators and losses to their respective initial values. */
    void resetEvaluators();

    /**
     * Gets the training {@link Loss} function of the trainer.
     *
     * @return the {@link Loss} function
     */
    Loss getLoss();

    /**
     * Gets the validation {@link Loss} function of the trainer.
     *
     * @return the validation {@link Loss} function
     */
    Loss getValidationLoss();

    /**
     * Returns the model used to create this trainer.
     *
     * @return the model associated with this trainer
     */
    Model getModel();

    /**
     * Gets all training {@link Evaluator}s.
     *
     * @return the training metrics
     */
    List<Evaluator> getTrainingEvaluators();

    /**
     * Gets the training {@link Evaluator} that is an instance of the given {@link Class}.
     *
     * @param clazz the {@link Class} of the {@link Evaluator} sought
     * @param <T> the type of the training evaluator
     * @return the training evaluator requested
     */
    <T extends Evaluator> T getTrainingEvaluator(Class<T> clazz);

    /**
     * Gets all validation {@link Evaluator}s.
     *
     * @return the validation metrics
     */
    List<Evaluator> getValidationEvaluators();

    /**
     * Gets the validation {@link Evaluator} that is an instance of the given {@link Class}.
     *
     * @param clazz the {@link Class} of the {@link Evaluator} sought
     * @param <T> the type of the validation evaluator
     * @return the validation evaluator requested
     */
    <T extends Evaluator> T getValidationEvaluator(Class<T> clazz);

    /**
     * Gets the {@link NDManager} from the model.
     *
     * @return the {@link NDManager}
     */
    NDManager getManager();

    /** {@inheritDoc} */
    @Override
    void close();
}

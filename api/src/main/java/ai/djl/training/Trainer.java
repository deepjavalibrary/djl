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
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.training.listener.EpochTrainingListener;
import ai.djl.training.listener.EvaluatorTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The {@code Trainer} interface provides a session for model training.
 *
 * <p>{@code Trainer} provides an easy, and manageable interface for training. {@code Trainer} is
 * not thread-safe.
 *
 * <p>See the tutorials on:
 *
 * <ul>
 *   <li><a
 *       href="https://github.com/deepjavalibrary/djl/blob/master/jupyter/tutorial/02_train_your_first_model.ipynb">Training
 *       your first model</a>
 *   <li><a
 *       href="https://github.com/deepjavalibrary/djl/blob/master/jupyter/transfer_learning_on_cifar10.ipynb">Training
 *       using transfer learning</a>
 *   <li><a
 *       href="https://github.com/deepjavalibrary/djl/blob/master/jupyter/load_mxnet_model.ipynb">Inference
 *       with an MXNet model</a>
 * </ul>
 *
 * @see <a href="http://docs.djl.ai/docs/development/memory_management.html">The guide on memory
 *     management</a>
 */
public class Trainer implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(Trainer.class);

    private Model model;
    private NDManager manager;
    private Metrics metrics;
    private List<TrainingListener> listeners;
    private Device[] devices;
    private ParameterStore parameterStore;
    private List<Evaluator> evaluators;
    private Loss loss;
    private ExecutorService executorService;

    private boolean gradientsChecked;

    /**
     * Creates an instance of {@code Trainer} with the given {@link Model} and {@link
     * TrainingConfig}.
     *
     * @param model the model the trainer will train on
     * @param trainingConfig the configuration used by the trainer
     */
    public Trainer(Model model, TrainingConfig trainingConfig) {
        this.model = model;
        manager = model.getNDManager().newSubManager();
        manager.setName("trainer");
        devices = trainingConfig.getDevices();
        loss = trainingConfig.getLossFunction();
        Objects.requireNonNull(loss, "You must specify a loss for the trainer");
        evaluators = new ArrayList<>(trainingConfig.getEvaluators());
        evaluators.add(loss); // track loss as an evaluator by default
        executorService = trainingConfig.getExecutorService();

        ParameterServer parameterServer =
                manager.getEngine().newParameterServer(trainingConfig.getOptimizer());

        parameterStore = new ParameterStore(manager, false);
        parameterStore.setParameterServer(parameterServer, devices);

        listeners = trainingConfig.getTrainingListeners();
        notifyListeners(listener -> listener.onTrainingBegin(this));
    }

    /**
     * Initializes the {@link Model} that the {@code Trainer} is going to train.
     *
     * @param shapes an array of {@code Shape} of the inputs
     */
    public void initialize(Shape... shapes) {
        model.getBlock().initialize(model.getNDManager(), model.getDataType(), shapes);
        // call getValue on all params to initialize on all devices
        model.getBlock()
                .getParameters()
                .forEach(
                        pair -> {
                            for (Device device : devices) {
                                parameterStore.getValue(pair.getValue(), device, true);
                            }
                        });
    }

    /**
     * Fetches an iterator that can iterate through the given {@link Dataset}.
     *
     * @param dataset the dataset to iterate through
     * @return an {@link Iterable} of {@link Batch} that contains batches of data from the dataset
     * @throws IOException for various exceptions depending on the dataset
     * @throws TranslateException if there is an error while processing input
     */
    public Iterable<Batch> iterateDataset(Dataset dataset) throws IOException, TranslateException {
        return dataset.getData(getManager(), executorService);
    }

    /**
     * Returns a new instance of {@link GradientCollector}.
     *
     * @return a new instance of {@link GradientCollector}
     */
    public GradientCollector newGradientCollector() {
        return manager.getEngine().newGradientCollector();
    }

    /**
     * Applies the forward function of the model once on the given input {@link NDList}.
     *
     * @param input the input {@link NDList}
     * @return the output of the forward function
     */
    public NDList forward(NDList input) {
        long begin = System.nanoTime();
        try {
            return model.getBlock().forward(parameterStore, input, true);
        } finally {
            addMetric("forward", begin);
        }
    }

    /**
     * Applies the forward function of the model once with both data and labels.
     *
     * @param data the input data {@link NDList}
     * @param labels the input labels {@link NDList}
     * @return the output of the forward function
     */
    public NDList forward(NDList data, NDList labels) {
        long begin = System.nanoTime();
        try {
            return model.getBlock().forward(parameterStore, data, labels, null);
        } finally {
            addMetric("forward", begin);
        }
    }

    /**
     * Evaluates function of the model once on the given input {@link NDList}.
     *
     * @param input the input {@link NDList}
     * @return the output of the predict function
     */
    public NDList evaluate(NDList input) {
        return model.getBlock().forward(parameterStore, input, false, null);
    }

    /** Updates all of the parameters of the model once. */
    public void step() {
        if (!gradientsChecked) {
            checkGradients();
        }

        long begin = System.nanoTime();
        parameterStore.updateAllParameters();
        addMetric("step", begin);
    }

    /**
     * Returns the Metrics param used for benchmarking.
     *
     * @return the the Metrics param used for benchmarking
     */
    public Metrics getMetrics() {
        return metrics;
    }

    /**
     * Attaches a Metrics param to use for benchmarking.
     *
     * @param metrics the Metrics class
     */
    public void setMetrics(Metrics metrics) {
        this.metrics = metrics;
    }

    /**
     * Returns the devices used for training.
     *
     * @return the devices used for training
     */
    public Device[] getDevices() {
        return devices;
    }

    /**
     * Gets the training {@link Loss} function of the trainer.
     *
     * @return the {@link Loss} function
     */
    public Loss getLoss() {
        return loss;
    }

    /**
     * Returns the model used to create this trainer.
     *
     * @return the model associated with this trainer
     */
    public Model getModel() {
        return model;
    }

    /**
     * Returns the {@link ExecutorService}.
     *
     * @return the {@link ExecutorService}
     */
    public Optional<ExecutorService> getExecutorService() {
        return Optional.ofNullable(executorService);
    }

    /**
     * Gets all {@link Evaluator}s.
     *
     * @return the evaluators used during training
     */
    public List<Evaluator> getEvaluators() {
        return evaluators;
    }

    /**
     * Executes a method on each of the {@link TrainingListener}s.
     *
     * @param listenerConsumer a consumer that executes the method
     */
    public void notifyListeners(Consumer<TrainingListener> listenerConsumer) {
        listeners.forEach(listenerConsumer);
    }

    /**
     * Returns the {@link TrainingResult}.
     *
     * @return the {@code TrainingResult}
     */
    public TrainingResult getTrainingResult() {
        TrainingResult result = new TrainingResult();
        for (TrainingListener listener : listeners) {
            if (listener instanceof EpochTrainingListener) {
                result.setEpoch(((EpochTrainingListener) listener).getNumEpochs());
            } else if (listener instanceof EvaluatorTrainingListener) {
                EvaluatorTrainingListener l = (EvaluatorTrainingListener) listener;
                result.setEvaluations(l.getLatestEvaluations());
            }
        }
        return result;
    }

    /**
     * Gets the {@link NDManager} from the model.
     *
     * @return the {@link NDManager}
     */
    public NDManager getManager() {
        return manager;
    }

    /** {@inheritDoc} */
    @SuppressWarnings("deprecation")
    @Override
    protected void finalize() throws Throwable {
        if (manager.isOpen()) {
            if (logger.isDebugEnabled()) {
                logger.warn("Trainer for {} was not closed explicitly.", model.getName());
            }
            close();
        }
        super.finalize();
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        notifyListeners(listener -> listener.onTrainingEnd(this));

        parameterStore.sync();
        manager.close();
    }

    /**
     * Checks if all gradients are zeros. This prevent users from calling step() without running
     * {@code backward}.
     */
    private void checkGradients() {
        List<NDArray> grads = new ArrayList<>();
        model.getBlock()
                .getParameters()
                .values()
                .stream()
                .filter(Parameter::requiresGradient)
                .forEach(
                        param ->
                                grads.add(
                                        parameterStore
                                                .getValue(param, devices[0], true)
                                                .getGradient()));

        try (NDManager scoped = manager.newSubManager()) {
            scoped.tempAttachAll(new NDList(grads));
            NDList list = new NDList(grads.stream().map(NDArray::sum).toArray(NDArray[]::new));
            float gradSum = NDArrays.stack(list).sum().getFloat();

            if (gradSum == 0f) {
                throw new IllegalStateException(
                        "Gradient values are all zeros, please call gradientCollector.backward() on"
                                + "your target NDArray (usually loss), before calling step() ");
            }

            gradientsChecked = true;
        }
    }

    /**
     * Helper to add a metric for a time difference.
     *
     * @param metricName the metric name
     * @param begin the time difference start (this method is called at the time difference end)
     */
    public void addMetric(String metricName, long begin) {
        if (metrics != null && begin > 0L) {
            metrics.addMetric(metricName, System.nanoTime() - begin);
        }
    }
}

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
package ai.djl.mxnet.engine;

import ai.djl.Device;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataDesc;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterServer;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.loss.Loss;
import ai.djl.training.metrics.TrainingMetrics;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.util.PairList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MxTrainer implements Trainer {

    private static final Logger logger = LoggerFactory.getLogger(MxTrainer.class);

    private MxModel model;
    private MxNDManager manager;
    // this is performance metrics similar to predictor metrics, not training accuracy or loss
    // currently not implemented
    private Metrics metrics;
    private Optimizer optimizer;
    private Device[] devices;
    private PairList<String, Parameter> parameters;
    private ParameterStore parameterStore;
    private Loss loss;

    private List<TrainingMetrics> trainingMetrics;
    private boolean gradientsChecked;

    MxTrainer(MxModel model, TrainingConfig trainingConfig) {
        this.model = model;
        this.manager = (MxNDManager) model.getNDManager().newSubManager();
        Block block = model.getBlock();
        optimizer = trainingConfig.getOptimizer();
        devices = trainingConfig.getDevices();
        loss = trainingConfig.getLossFunction();
        trainingMetrics = trainingConfig.getTrainingMetrics();
        parameters = block.getParameters();
        if (devices.length > 1) {
            parameterStore = new ParameterStore(parameters, devices);
            parameters
                    .stream()
                    .forEach(param -> param.getValue().setParameterStore(parameterStore));
        }
    }

    @Override
    public void initialize(DataDesc[] inputDescriptor) {
        Shape[] shapes =
                Arrays.stream(inputDescriptor).map(DataDesc::getShape).toArray(Shape[]::new);
        model.getBlock().initialize(manager, model.getDataType(), devices, shapes);
    }

    @Override
    public GradientCollector newGradientCollector() {
        return new MxGradientCollector();
    }

    /** {@inheritDoc} */
    @Override
    public ParameterServer newParameterServer(Optimizer optimizer) {
        return new MxParameterServer(optimizer);
    }

    @Override
    public NDList forward(NDList input) {
        return model.getBlock().forward(input);
    }

    @Override
    public NDArray loss(NDList labels, NDList preds) {
        NDArray l = loss.update(labels, preds);
        trainingMetrics.forEach(metric -> metric.update(labels, preds));
        return l;
    }

    /** {@inheritDoc} */
    @Override
    public void step() {
        if (optimizer == null) {
            throw new IllegalStateException(
                    "No optimizer is set for trainer, please initialize"
                            + "your trainer with an Optimizer.");
        }
        if (!gradientsChecked) {
            checkGradients();
        }
        if (parameterStore != null) {
            parameterStore.updateAllParameters();
        } else {
            optimizer.updateAllParameters(parameters);
        }
    }

    @Override
    public void setMetrics(Metrics metrics) {
        this.metrics = metrics;
    }

    @Override
    public void resetTrainingMetrics() {
        loss.reset();
        trainingMetrics.forEach(TrainingMetrics::reset);
    }

    @Override
    public float getLoss() {
        return loss.getMetric().getValue();
    }

    public Metrics getMetrics() {
        return metrics;
    }

    @Override
    public List<TrainingMetrics> getTrainingMetrics() {
        return trainingMetrics;
    }

    @Override
    public NDManager getManager() {
        return manager;
    }

    /**
     * Check if all gradients are zeros, prevent users from calling step() without running {@code
     * backward}.
     */
    private void checkGradients() {
        List<NDArray> grads = new ArrayList<>();
        if (parameterStore != null) {
            parameterStore.setParameterServer(newParameterServer(optimizer));
            parameters
                    .values()
                    .stream()
                    .filter(Parameter::requireGradient)
                    .forEach(param -> grads.add(parameterStore.getValue(param, devices[0])));
        } else {

            grads.addAll(
                    parameters
                            .values()
                            .stream()
                            .filter(Parameter::requireGradient)
                            .map(param -> param.getArray().getGradient())
                            .collect(Collectors.toList()));
        }
        NDArray gradSum =
                NDArrays.stack(
                        new NDList(grads.stream().map(NDArray::sum).toArray(NDArray[]::new)));
        float[] sums = gradSum.sum().toFloatArray();
        gradSum.close();
        float sum = 0f;
        for (float num : sums) {
            sum += num;
        }
        if (sum == 0f) {
            throw new IllegalStateException(
                    "Gradient values are all zeros, please call gradientCollector.backward() on"
                            + "your target NDArray (usually loss), before calling step() ");
        }

        gradientsChecked = true;
    }

    /** {@inheritDoc} */
    @SuppressWarnings("deprecation")
    @Override
    protected void finalize() throws Throwable {
        if (manager.isOpen()) {
            if (logger.isDebugEnabled()) {
                logger.warn("Model was not closed explicitly: {}", getClass().getSimpleName());
            }
            close();
        }
        super.finalize();
    }

    @Override
    public void close() {
        if (parameterStore != null) {
            parameterStore.close();
        }
        manager.close();
    }
}

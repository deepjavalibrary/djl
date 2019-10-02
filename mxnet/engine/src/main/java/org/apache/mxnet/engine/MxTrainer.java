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
package org.apache.mxnet.engine;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.ai.Device;
import software.amazon.ai.metric.Metrics;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.nn.Block;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.training.GradientCollector;
import software.amazon.ai.training.ParameterServer;
import software.amazon.ai.training.ParameterStore;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.training.TrainingConfig;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.util.PairList;

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
    private boolean gradientsChecked;

    MxTrainer(MxModel model, TrainingConfig trainingConfig) {
        this.model = model;
        this.manager = (MxNDManager) model.getNDManager().newSubManager();
        Block block = model.getBlock();
        optimizer = trainingConfig.getOptimizer();
        devices = trainingConfig.getDevices();
        parameters = block.getParameters();
        if (devices.length > 1) {
            parameterStore = new ParameterStore(parameters, devices);
            parameters
                    .stream()
                    .forEach(param -> param.getValue().setParameterStore(parameterStore));
        }
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
            parameterStore.updateAllParameters(optimizer);
        } else {
            optimizer.updateAllParameters(parameters);
        }
    }

    @Override
    public void setMetrics(Metrics metrics) {
        this.metrics = metrics;
    }

    public Metrics getMetrics() {
        return metrics;
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
            for (Parameter parameter : parameters.values()) {
                // only check one device
                grads.add(parameterStore.getValue(parameter, devices[0]));
            }
        } else {

            grads.addAll(
                    parameters
                            .stream()
                            .map(pair -> pair.getValue().getArray().getGradient())
                            .collect(Collectors.toList()));
        }
        NDArray gradSum =
                NDArrays.stack(
                        new NDList(grads.stream().map(NDArray::sum).toArray(NDArray[]::new)));
        float[] sums = gradSum.sum().toFloatArray();
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

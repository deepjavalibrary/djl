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
package ai.djl.serving.wlm;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.ZooModel;
import java.nio.file.Path;

/** A class represent a loaded model and it's metadata. */
public class ModelInfo implements AutoCloseable {

    private String modelName;
    private String modelUrl;

    private int minWorkers;
    private int maxWorkers;
    private int queueSize;
    private int batchSize;
    private int maxBatchDelay;
    private long maxIdleTime;

    private ZooModel<Input, Output> model;

    /**
     * Constructs a new {@code ModelInfo} instance.
     *
     * @param modelName the name of the model that will be used as HTTP endpoint
     * @param modelUrl the model url
     * @param model the {@link ZooModel}
     * @param queueSize the maximum request queue size
     */
    public ModelInfo(
            String modelName, String modelUrl, ZooModel<Input, Output> model, int queueSize) {
        this.modelName = modelName;
        this.modelUrl = modelUrl;
        this.model = model;
        batchSize = 1;
        maxBatchDelay = 100;
        // TODO make this configurable
        this.maxIdleTime = 60; // default max idle time 60s
        this.queueSize = queueSize;
    }

    /**
     * Returns the loaded {@link ZooModel}.
     *
     * @return the loaded {@link ZooModel}
     */
    public ZooModel<Input, Output> getModel() {
        return model;
    }

    /**
     * Returns the model name.
     *
     * @return the model name
     */
    public String getModelName() {
        return modelName;
    }

    /**
     * Returns the model url.
     *
     * @return the model url
     */
    public String getModelUrl() {
        return modelUrl;
    }

    /**
     * Returns the model cache directory.
     *
     * @return the model cache directory
     */
    public Path getModelDir() {
        return model.getModelPath();
    }

    /**
     * returns the configured maxIdleTime of workers.
     *
     * @return the maxIdleTime
     */
    public long getMaxIdleTime() {
        return maxIdleTime;
    }

    /**
     * set the configured maxIdleTime of workers.
     *
     * @param maxIdleTime the maxIdleTime to set
     */
    public void setMaxIdleTime(long maxIdleTime) {
        this.maxIdleTime = maxIdleTime;
    }

    /**
     * Returns the configured minimum number of workers.
     *
     * @return the configured minimum number of workers
     */
    public int getMinWorkers() {
        return minWorkers;
    }

    /**
     * Sets the minimum number of workers.
     *
     * @param minWorkers the minimum number of workers
     */
    public void setMinWorkers(int minWorkers) {
        this.minWorkers = minWorkers;
    }

    /**
     * Returns the configured maximum number of workers.
     *
     * @return the configured maximum number of workers
     */
    public int getMaxWorkers() {
        return maxWorkers;
    }

    /**
     * Sets the maximum number of workers.
     *
     * @param maxWorkers the maximum number of workers
     */
    public void setMaxWorkers(int maxWorkers) {
        this.maxWorkers = maxWorkers;
    }

    /**
     * Returns the configured batch size.
     *
     * @return the configured batch size
     */
    public int getBatchSize() {
        return batchSize;
    }

    /**
     * Sets the batch size.
     *
     * @param batchSize the batch size
     */
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    /**
     * Returns the maximum delay in milliseconds to aggregate a batch.
     *
     * @return the maximum delay in milliseconds to aggregate a batch
     */
    public int getMaxBatchDelay() {
        return maxBatchDelay;
    }

    /**
     * Sets the maximum delay in milliseconds to aggregate a batch.
     *
     * @param maxBatchDelay the maximum delay in milliseconds to aggregate a batch
     */
    public void setMaxBatchDelay(int maxBatchDelay) {
        this.maxBatchDelay = maxBatchDelay;
    }

    /**
     * returns the configured size of the workers queue.
     *
     * @return requested size of the workers queue.
     */
    public int getQueueSize() {
        return queueSize;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (model != null) {
            model.close();
        }
    }
}

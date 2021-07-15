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
import ai.djl.repository.FilenameUtils;
import ai.djl.repository.zoo.ZooModel;
import java.net.URI;
import java.nio.file.Path;
import java.util.Objects;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A class represent a loaded model and it's metadata. */
public final class ModelInfo implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(ModelInfo.class);

    private String modelName;
    private String version;
    private String modelUrl;

    private int minWorkers;
    private int maxWorkers;
    private int queueSize;
    private int batchSize;
    private int maxBatchDelay;
    private int maxIdleTime;

    private ZooModel<Input, Output> model;

    /**
     * Constructs a new {@code ModelInfo} instance.
     *
     * @param modelName the name of the model that will be used as HTTP endpoint
     * @param version the version of the model
     * @param modelUrl the model url
     * @param model the {@link ZooModel}
     * @param queueSize the maximum request queue size
     * @param maxIdleTime the initial maximum idle time for workers.
     * @param maxBatchDelay the initial maximum delay when scaling up before giving up.
     * @param batchSize the batch size for this model.
     */
    public ModelInfo(
            String modelName,
            String version,
            String modelUrl,
            ZooModel<Input, Output> model,
            int queueSize,
            int maxIdleTime,
            int maxBatchDelay,
            int batchSize) {
        this.modelName = modelName;
        this.version = version;
        this.modelUrl = modelUrl;
        this.model = model;
        this.maxBatchDelay = maxBatchDelay;
        this.maxIdleTime = maxIdleTime; // default max idle time 60s
        this.queueSize = queueSize;
        this.batchSize = batchSize;
    }

    /**
     * Sets a new batchSize and returns a new configured ModelInfo object. You have to
     * triggerUpdates in the {@code ModelManager} using this new model.
     *
     * @param batchSize the batchSize to set
     * @param maxBatchDelay maximum time to wait for a free space in worker queue after scaling up
     *     workers before giving up to offer the job to the queue.
     * @return new configured ModelInfo.
     */
    public ModelInfo configureModelBatch(int batchSize, int maxBatchDelay) {
        this.batchSize = batchSize;
        this.maxBatchDelay = maxBatchDelay;
        return this;
    }

    /**
     * Sets new workers capcities for this model and returns a new configured ModelInfo object. You
     * have to triggerUpdates in the {@code ModelManager} using this new model.
     *
     * @param minWorkers minimum amount of workers.
     * @param maxWorkers maximum amount of workers.
     * @return new configured ModelInfo.
     */
    public ModelInfo scaleWorkers(int minWorkers, int maxWorkers) {
        this.minWorkers = minWorkers;
        this.maxWorkers = maxWorkers;
        return this;
    }

    /**
     * Sets new configuration for the workerPool backing this model and returns a new configured
     * ModelInfo object. You have to triggerUpdates in the {@code ModelManager} using this new
     * model.
     *
     * @param maxIdleTime time a WorkerThread can be idle before scaling down this worker.
     * @return new configured ModelInfo.
     */
    public ModelInfo configurePool(int maxIdleTime) {
        this.maxIdleTime = maxIdleTime;
        return this;
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
     * Returns the model version.
     *
     * @return the model version
     */
    public String getVersion() {
        return version;
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
    public int getMaxIdleTime() {
        return maxIdleTime;
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
     * Returns the configured maximum number of workers.
     *
     * @return the configured maximum number of workers
     */
    public int getMaxWorkers() {
        return maxWorkers;
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
     * Returns the maximum delay in milliseconds to aggregate a batch.
     *
     * @return the maximum delay in milliseconds to aggregate a batch
     */
    public int getMaxBatchDelay() {
        return maxBatchDelay;
    }

    /**
     * Returns the configured size of the workers queue.
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
            logger.debug("closing model {}", modelName);
            model.close();
        }
    }

    /**
     * Infer model name form model URL in case model name is not provided.
     *
     * @param url the model URL
     * @return the model name
     */
    public static String inferModelNameFromUrl(String url) {
        URI uri = URI.create(url);
        String path = uri.getPath();
        boolean isDirectory = path.endsWith("/");
        if (isDirectory) {
            path = path.substring(0, path.length() - 1);
        }
        int pos = path.lastIndexOf('/');
        String modelName;
        if (pos >= 0) {
            modelName = path.substring(pos + 1);
        } else {
            modelName = path;
        }
        if (!isDirectory) {
            modelName = FilenameUtils.getNamePart(modelName);
        }
        modelName = modelName.replaceAll("(\\W|^_)", "_");
        return modelName;
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof ModelInfo)) {
            return false;
        }
        ModelInfo modelInfo = (ModelInfo) o;
        return modelName.equals(modelInfo.modelName) && Objects.equals(version, modelInfo.version);
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return Objects.hash(modelName, version);
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        if (version != null) {
            return modelName + ':' + version;
        }
        return modelName;
    }
}

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

import ai.djl.ModelException;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.serving.http.BadRequestException;
import ai.djl.serving.http.DescribeModelResponse;
import ai.djl.serving.util.ConfigManager;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A class that in charge of managing models. */
public final class ModelManager {

    private static final Logger logger = LoggerFactory.getLogger(ModelManager.class);

    private static ModelManager modelManager;

    private ConfigManager configManager;
    private WorkLoadManager wlm;
    private ConcurrentHashMap<String, ModelInfo> models;
    private Set<String> startupModels;

    private ModelManager(ConfigManager configManager) {
        this.configManager = configManager;
        wlm = new WorkLoadManager(configManager);
        models = new ConcurrentHashMap<>();
        startupModels = new HashSet<>();
    }

    /**
     * Initialized the global {@code ModelManager} instance.
     *
     * @param configManager the configuration
     */
    public static void init(ConfigManager configManager) {
        modelManager = new ModelManager(configManager);
    }

    /**
     * Returns the singleton {@code ModelManager} instance.
     *
     * @return the singleton {@code ModelManager} instance
     */
    public static ModelManager getInstance() {
        return modelManager;
    }

    /**
     * Registers and loads a model.
     *
     * @param modelName the name of the model for HTTP endpoint
     * @param modelUrl the model url
     * @param engineName the engine to load the model
     * @param batchSize the batch size
     * @param maxBatchDelay the maximum delay for batching
     * @param maxIdleTime the maximum idle time of the worker threads before scaling down.
     * @return a {@code CompletableFuture} instance
     */
    public CompletableFuture<ModelInfo> registerModel(
            final String modelName,
            final String modelUrl,
            final String engineName,
            final int batchSize,
            final int maxBatchDelay,
            final int maxIdleTime) {
        return CompletableFuture.supplyAsync(
                () -> {
                    try {
                        Criteria<Input, Output> criteria =
                                Criteria.builder()
                                        .setTypes(Input.class, Output.class)
                                        .optModelUrls(modelUrl)
                                        .optEngine(engineName)
                                        .build();
                        ZooModel<Input, Output> model = ModelZoo.loadModel(criteria);
                        ModelInfo modelInfo =
                                new ModelInfo(
                                        modelName,
                                        modelUrl,
                                        model,
                                        configManager.getJobQueueSize(),
                                        maxIdleTime,
                                        maxBatchDelay,
                                        batchSize);

                        ModelInfo existingModel = models.putIfAbsent(modelName, modelInfo);
                        if (existingModel != null) {
                            // model already exists
                            model.close();
                            throw new BadRequestException(
                                    "Model " + modelName + " is already registered.");
                        }
                        logger.info("Model {} loaded.", modelName);

                        return modelInfo;
                    } catch (ModelException | IOException e) {
                        throw new CompletionException(e);
                    }
                });
    }

    /**
     * Unregisters a model by its name.
     *
     * @param modelName the model name to be unregistered
     * @return {@code true} if unregister success
     */
    public boolean unregisterModel(String modelName) {
        ModelInfo model = models.remove(modelName);
        if (model == null) {
            logger.warn("Model not found: " + modelName);
            return false;
        }
        model = model.scaleWorkers(0, 0);
        wlm.modelChanged(model);
        startupModels.remove(modelName);
        model.close();
        logger.info("Model {} unregistered.", modelName);
        return true;
    }

    /**
     * trigger that a ModelInfo has been updated. Updates model workers for this model and scales
     * up/down all workers to match the parameters for the model.
     *
     * @param modelInfo the model that has been updated
     */
    public void triggerModelUpdated(ModelInfo modelInfo) {
        if (!models.containsKey(modelInfo.getModelName())) {
            throw new AssertionError("Model not found: " + modelInfo.getModelName());
        }
        logger.debug("updateModel: {}", modelInfo.getModelName());
        models.put(modelInfo.getModelName(), modelInfo);
        wlm.modelChanged(modelInfo);
    }

    /**
     * Returns the registry of all models.
     *
     * @return the registry of all models
     */
    public Map<String, ModelInfo> getModels() {
        return models;
    }

    /**
     * Returns a set of models that was loaded at startup.
     *
     * @return a set of models that was loaded at startup
     */
    public Set<String> getStartupModels() {
        return startupModels;
    }

    /**
     * Adds an inference job to the job queue. Assign the job to the next free worker.
     *
     * @param job an inference job to be executed
     * @return {@code true} if submit success
     * @throws ModelNotFoundException if the model is not registered
     */
    public boolean addJob(Job job) throws ModelNotFoundException {
        String modelName = job.getModelName();
        ModelInfo model = models.get(modelName);
        if (model == null) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        }
        return wlm.addJob(model, job);
    }

    /**
     * Returns a list of worker information for specified model.
     *
     * @param modelName the model to be queried
     * @return a list of worker information for specified model
     * @throws ModelNotFoundException if specified model not found
     */
    public DescribeModelResponse describeModel(String modelName) throws ModelNotFoundException {
        ModelInfo model = models.get(modelName);
        if (model == null) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        }

        DescribeModelResponse resp = new DescribeModelResponse();
        resp.setModelName(modelName);
        resp.setModelUrl(model.getModelUrl());
        resp.setBatchSize(model.getBatchSize());
        resp.setMaxBatchDelay(model.getMaxBatchDelay());
        resp.setMaxWorkers(model.getMaxWorkers());
        resp.setMinWorkers(model.getMinWorkers());
        resp.setMaxIdleTime(model.getMaxIdleTime());
        resp.setLoadedAtStartup(startupModels.contains(modelName));

        int activeWorker = wlm.getNumRunningWorkers(modelName);
        int targetWorker = model.getMinWorkers();
        resp.setStatus(activeWorker >= targetWorker ? "Healthy" : "Unhealthy");

        List<WorkerThread> workers = wlm.getWorkers(modelName);
        for (WorkerThread worker : workers) {
            int workerId = worker.getWorkerId();
            long startTime = worker.getStartTime();
            boolean isRunning = worker.isRunning();
            int gpuId = worker.getGpuId();
            resp.addWorker(workerId, startTime, isRunning, gpuId);
        }
        return resp;
    }

    /**
     * Sends model server health status to client.
     *
     * @return completableFuture with eventually result in the future after async execution
     */
    public CompletableFuture<String> workerStatus() {
        return CompletableFuture.supplyAsync(
                () -> {
                    String response = "Healthy";
                    int numWorking = 0;

                    int numScaled = 0;
                    for (Map.Entry<String, ModelInfo> m : models.entrySet()) {
                        numScaled += m.getValue().getMinWorkers();
                        numWorking += wlm.getNumRunningWorkers(m.getValue().getModelName());
                    }

                    if ((numWorking > 0) && (numWorking < numScaled)) {
                        response = "Partial Healthy";
                    } else if ((numWorking == 0) && (numScaled > 0)) {
                        response = "Unhealthy";
                    }

                    return response;
                });
    }
}

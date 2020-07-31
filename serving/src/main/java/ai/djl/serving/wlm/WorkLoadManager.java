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

import ai.djl.serving.util.ConfigManager;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

class WorkLoadManager {

    private ConfigManager configManager;
    private AtomicInteger gpuCounter;
    private ExecutorService threadPool;
    private ConcurrentHashMap<String, List<WorkerThread>> workers;

    public WorkLoadManager(ConfigManager configManager) {
        this.configManager = configManager;
        this.gpuCounter = new AtomicInteger(0);
        threadPool = Executors.newCachedThreadPool();
        workers = new ConcurrentHashMap<>();
    }

    public List<WorkerThread> getWorkers(String modelName) {
        List<WorkerThread> list = workers.get(modelName);
        if (list == null) {
            return Collections.emptyList();
        }
        return list;
    }

    public boolean hasWorker(String modelName) {
        List<WorkerThread> worker = workers.get(modelName);
        return worker != null && !worker.isEmpty();
    }

    public int getNumRunningWorkers(String modelName) {
        int numWorking = 0;
        List<WorkerThread> threads = workers.get(modelName);
        if (threads != null) {
            for (WorkerThread thread : threads) {
                if ((thread.getState() != WorkerState.WORKER_STOPPED)
                        && (thread.getState() != WorkerState.WORKER_ERROR)
                        && (thread.getState() != WorkerState.WORKER_SCALED_DOWN)) {
                    ++numWorking;
                }
            }
        }
        return numWorking;
    }

    public void modelChanged(ModelInfo modelInfo) {
        synchronized (modelInfo.getModelName()) {
            int minWorker = modelInfo.getMinWorkers();
            int maxWorker = modelInfo.getMaxWorkers();
            List<WorkerThread> threads;
            if (minWorker == 0) {
                threads = workers.remove(modelInfo.getModelName());
                if (threads == null) {
                    return;
                }
            } else {
                threads = workers.computeIfAbsent(modelInfo.getModelName(), k -> new ArrayList<>());
            }

            int currentWorkers = threads.size();
            if (currentWorkers < minWorker) {
                addThreads(threads, modelInfo, minWorker - currentWorkers);
            } else {
                for (int i = currentWorkers - 1; i >= maxWorker; --i) {
                    WorkerThread thread = threads.remove(i);
                    thread.shutdown();
                }
            }
        }
    }

    public void scheduleAsync(Runnable r) {
        threadPool.execute(r);
    }

    private void addThreads(List<WorkerThread> threads, ModelInfo model, int count) {
        int maxGpu = configManager.getNumberOfGpu();
        for (int i = 0; i < count; ++i) {
            int gpuId = -1;

            if (maxGpu > 0) {
                gpuId = gpuCounter.accumulateAndGet(maxGpu, (prev, maxGpuId) -> ++prev % maxGpuId);
            }

            BatchAggregator aggregator = new BatchAggregator(model);
            WorkerThread thread = new WorkerThread(gpuId, model, aggregator);
            threads.add(thread);
            threadPool.submit(thread);
        }
    }
}

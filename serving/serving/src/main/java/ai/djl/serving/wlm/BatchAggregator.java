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
import io.netty.handler.codec.http.HttpResponseStatus;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.LinkedBlockingDeque;

/**
 * abstract class for all BatchAggregators. A batch aggregator check working queue and combines
 * multiple job into one batch. batches of jobs are used cause optimisations in separate engines.
 *
 * @author erik.bamberg@web.de
 */
abstract class BatchAggregator {

    protected int batchSize;
    protected List<Job> jobs;
    protected LinkedBlockingDeque<Job> jobQueue;

    /**
     * Constructs a new {@code BbatchAggregator} instance.
     *
     * @param model the model to use.
     * @param jobQueue the job queue for polling data from.
     */
    public BatchAggregator(ModelInfo model, LinkedBlockingDeque<Job> jobQueue) {
        this.batchSize = model.getBatchSize();
        this.jobQueue = jobQueue;
        jobs = new ArrayList<>();
    }

    /**
     * Poll the queue and return a list of Input Objects for the model.
     *
     * @return list of input objects to pass to the model.
     * @throws InterruptedException if thread gets interrupted while waiting for new data in the
     *     queue.
     */
    public List<Input> getRequest() throws InterruptedException {
        jobs = pollBatch();
        List<Input> list = new ArrayList<>(jobs.size());
        for (Job job : jobs) {
            job.setScheduled();
            list.add(job.getInput());
        }
        return list;
    }

    /**
     * Sends to response to all waiting clients.
     *
     * @param outputs list of model-outputs in same order as the input objects.
     */
    public void sendResponse(List<Output> outputs) {
        if (jobs.size() != outputs.size()) {
            throw new IllegalStateException("Not all jobs get response.");
        }

        int i = 0;
        for (Output output : outputs) {
            String requestId = output.getRequestId();
            Job job = jobs.get(i++);
            if (!job.getRequestId().equals(requestId)) {
                throw new IllegalStateException("Request response mismatched.");
            }
            job.sendOutput(output);
        }
        jobs.clear();
    }

    /**
     * Sends an error response to client.
     *
     * @param status the HTTP status
     * @param error the exception
     */
    public void sendError(HttpResponseStatus status, Throwable error) {
        for (Job job : jobs) {
            job.sendError(status, error);
        }
        jobs.clear();
    }

    /**
     * Fills in the list with a batch of jobs.
     *
     * @return a list of jobs read by this batch interation.
     * @throws InterruptedException if interrupted
     */
    protected abstract List<Job> pollBatch() throws InterruptedException;

    /**
     * Checks if this {@code BatchAggregator} and the thread can be shutdown or if this aggregator
     * waits for more data.
     *
     * @return true if we can shutdown the thread. for example when max idle time exceeded in
     *     temporary batch aggregator.
     */
    public abstract boolean isFinished();
}

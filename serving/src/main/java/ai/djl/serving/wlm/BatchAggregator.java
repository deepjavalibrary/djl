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
    }

    /**
     * Poll the queue and return a list of Input Objects for the model.
     *
     * @return list of input objects to pass to the model.
     * @throws InterruptedException if thread gets interrupted while waiting for new data in the
     *     queue.
     */
    public List<Job> getRequest() throws InterruptedException {
        List<Job> jobs = pollBatch();
        List<Job> list = new ArrayList<>(jobs.size());
        for (Job job : jobs) {
            job.setScheduled();
            list.add(job);
        }
        return list;
    }

    /**
     * Sends an internal server error to all jobs that are processed by this batch.
     *
     * @param jobs to send error.
     */
    public void sendError(List<Job> jobs) {
        for (Job job : jobs) {
            job.sendError(HttpResponseStatus.INTERNAL_SERVER_ERROR, "Internal server error");
        }
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

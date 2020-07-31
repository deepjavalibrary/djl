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

class BatchAggregator {

    private ModelInfo model;
    private List<Job> jobs;

    public BatchAggregator(ModelInfo model) {
        this.model = model;
        jobs = new ArrayList<>();
    }

    public List<Input> getRequest() throws InterruptedException {
        model.pollBatch(jobs);

        List<Input> list = new ArrayList<>(jobs.size());
        for (Job job : jobs) {
            job.setScheduled();
            list.add(job.getInput());
        }
        return list;
    }

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

    public void sendError() {
        for (Job job : jobs) {
            job.sendError(HttpResponseStatus.INTERNAL_SERVER_ERROR, "Internal server error");
        }
        jobs.clear();
    }
}

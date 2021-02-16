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
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A class represents an inference job. */
public class Job<T,U> {

    private static final Logger logger = LoggerFactory.getLogger(Job.class);

    private Consumer<U> callback;
    private BiConsumer<HttpResponseStatus,String> onError;
    private String modelName;
    private T input;
    private long begin;
    private long scheduled;

    /**
     * Constructs an new {@code Job} instance.
     *
     * @param ctx the {@code ChannelHandlerContext}
     * @param modelName the model name
     * @param input the input data
     * @param callback a callback function which is called after the output response is available
     */
    public Job(String modelName, T input,Consumer<U> callback, BiConsumer<HttpResponseStatus,String> onError) {
        this.callback = callback;
        this.onError = onError;
        this.modelName = modelName;
        this.input = input;

        begin = System.currentTimeMillis();
        scheduled = begin;
    }


    /**
     * Returns the model name that associated with this job.
     *
     * @return the model name that associated with this job
     */
    public String getModelName() {
        return modelName;
    }

    /**
     * Returns the input data.
     *
     * @return the input data
     */
    public T getInput() {
        return input;
    }

    /** Marks the job has been scheduled. */
    public void setScheduled() {
        scheduled = System.currentTimeMillis();
    }

    /**
     * Sends the response back to the client.
     *
     * @param output the output
     */
    public void sendOutput(U output) {
	
	if (callback!=null) {
	    callback.accept(output);
	}
	
        logger.debug(
                "Waiting time: {}, Backend time: {}",
                scheduled - begin,
                System.currentTimeMillis() - scheduled);
    }

    /**
     * Sends error to the client.
     *
     * @param status the HTTP status
     * @param error the error message
     */
    public void sendError(HttpResponseStatus status, String error) {
	if (onError!=null) {
	    onError.accept(status,error);
	} else {
	    logger.error(error);
	}
	
        logger.debug(
                "Waiting time: {}, Inference time: {}",
                scheduled - begin,
                System.currentTimeMillis() - begin);
    }
}

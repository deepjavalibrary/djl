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
import ai.djl.serving.http.InternalServerException;
import ai.djl.serving.util.NettyUtils;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpVersion;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A class represents an inference job. */
public class Job<T,U> {

    private static final Logger logger = LoggerFactory.getLogger(Job.class);

    private ChannelHandlerContext ctx;

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
     */
    public Job(ChannelHandlerContext ctx, String modelName, T input) {
        this.ctx = ctx;
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
        FullHttpResponse resp =
                new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.OK, false);
        
        if (output instanceof Output) {
            Output out=(Output)output;
            for (Map.Entry<String, String> entry : out.getProperties().entrySet()) {
                resp.headers().set(entry.getKey(), entry.getValue());
            }
            resp.content().writeBytes(out.getContent());
        } else {
          //  resp.content().writeBytes(null);
        }

        /*
         * We can load the models based on the configuration file.Since this Job is
         * not driven by the external connections, we could have a empty context for
         * this job. We shouldn't try to send a response to ctx if this is not triggered
         * by external clients.
         */
        if (ctx != null) {
            NettyUtils.sendHttpResponse(ctx, resp, true);
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
        /*
         * We can load the models based on the configuration file.Since this Job is
         * not driven by the external connections, we could have a empty context for
         * this job. We shouldn't try to send a response to ctx if this is not triggered
         * by external clients.
         */
        if (ctx != null) {
            NettyUtils.sendError(ctx, status, new InternalServerException(error));
        }

        logger.debug(
                "Waiting time: {}, Inference time: {}",
                scheduled - begin,
                System.currentTimeMillis() - begin);
    }
}

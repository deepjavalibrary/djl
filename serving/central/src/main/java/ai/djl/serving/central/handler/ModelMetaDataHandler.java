/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.central.handler;

import ai.djl.Application;
import ai.djl.repository.Artifact;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.serving.plugins.RequestHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * handler to handler model meta data requests.
 *
 * @author erik.bamberg@web.de
 */
public class ModelMetaDataHandler
        implements RequestHandler<CompletableFuture<Map<Application, List<Artifact>>>> {

    private static final Logger logger = LoggerFactory.getLogger(ModelMetaDataHandler.class);

    /**
     * chain of responsibility accept method.
     *
     * @param msg the request message to handle.
     * @return true/false if this handler can handler this kind of request.
     */
    @Override
    public boolean acceptInboundMessage(Object msg) {
        FullHttpRequest request = (FullHttpRequest) msg;

        String uri = request.uri();
        return uri.startsWith("/modelzoo/models");
    }

    /**
     * handle get Model meta data requests.
     *
     * @param ctx the context
     * @param req the full request
     * @param decoder query string decoder for this request.
     * @param segments parsed segments of the URL.
     */
    @Override
    public CompletableFuture<Map<Application, List<Artifact>>> handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments) {
        logger.info("request models");
        return CompletableFuture.supplyAsync(
                () -> {
                    try {
                        return ModelZoo.listModels();
                    } catch (IOException | ModelNotFoundException ex) {
                        throw new IllegalArgumentException(ex.getMessage(), ex);
                    }
                });
    }
}

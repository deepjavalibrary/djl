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
package ai.djl.repository.handler;

import ai.djl.Application;
import ai.djl.repository.Artifact;
import ai.djl.repository.responseencoder.JsonResponse;
import ai.djl.repository.zoo.ModelZoo;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.FullHttpRequest;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * handler to handler model meta data requests.
 *
 * @author erik.bamberg@web.de
 */
public class ModelMetaDataHandler extends SimpleChannelInboundHandler<FullHttpRequest> {

    private JsonResponse jsonResponse;

    /** construct handler. */
    public ModelMetaDataHandler() {
        jsonResponse = new JsonResponse();
    }

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
        return uri.startsWith("/models");
    }

    /**
     * handle get Model meta data requests.
     *
     * @param ctx the context
     * @param request the full request
     * @throws Exception any exception during execution.
     */
    @Override
    protected void channelRead0(ChannelHandlerContext ctx, FullHttpRequest request)
            throws Exception {

        CompletableFuture.supplyAsync(
                        () -> {
                            try {
                                return ModelZoo.listModels();
                            } catch (Exception ex) {
                                throw new IllegalArgumentException(ex.getMessage(), ex);
                            }
                        })
                .exceptionally((ex) -> new HashMap<Application, List<Artifact>>())
                .thenAccept(
                        modelMap -> {
                            jsonResponse.send(ctx, request, modelMap);
                        });
    }
}

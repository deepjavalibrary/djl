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

import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.serving.central.responseencoder.HttpRequestResponse;
import ai.djl.serving.central.utils.ModelUri;
import ai.djl.serving.central.utils.NettyUtils;
import ai.djl.serving.http.BadRequestException;
import ai.djl.serving.plugins.RequestHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.io.IOException;
import java.net.URI;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * A handler to handle download requests from the ModelView.
 *
 * @author anfee1@morgan.edu
 */
public class ModelDownloadHandler implements RequestHandler<CompletableFuture<Map<String, URI>>> {

    HttpRequestResponse jsonResponse;

    /** Constructs a ModelDownloadHandler. */
    public ModelDownloadHandler() {
        jsonResponse = new HttpRequestResponse();
    }

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) {
        FullHttpRequest request = (FullHttpRequest) msg;

        String uri = request.uri();
        return uri.startsWith("/serving/models?");
    }

    /**
     * Handles the deployment request by forwarding the request to the serving-instance.
     *
     * @param ctx the context
     * @param req the full request
     * @param decoder the queryStringDecoder
     * @param segments of the parsed URL
     */
    @Override
    public CompletableFuture<Map<String, URI>> handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments) {
        String modelName = NettyUtils.getParameter(decoder, "modelName", null);
        String modelGroupId = NettyUtils.getParameter(decoder, "groupId", null);
        String modelArtifactId = NettyUtils.getParameter(decoder, "artifactId", null);
        return CompletableFuture.supplyAsync(
                        () -> {
                            try {
                                if (modelName != null) {
                                    return ModelUri.uriFinder(
                                            modelArtifactId, modelGroupId, modelName);
                                } else {
                                    throw new BadRequestException("modelName is mandatory.");
                                }

                            } catch (IOException | ModelNotFoundException ex) {
                                throw new IllegalArgumentException(ex.getMessage(), ex);
                            }
                        })
                .exceptionally((ex) -> Collections.emptyMap());
    }
}

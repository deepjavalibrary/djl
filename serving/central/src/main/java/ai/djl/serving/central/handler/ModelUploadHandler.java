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

import ai.djl.serving.central.http.BadRequestException;
import ai.djl.serving.central.responseencoder.HttpRequestResponse;
import ai.djl.serving.central.utils.NettyUtils;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

/** A handler to handle upload requests from the ModelView. */
public class ModelUploadHandler extends SimpleChannelInboundHandler<FullHttpRequest> {

    HttpRequestResponse jsonResponse;

    /** Constructs a ModelUploadHandler. */
    public ModelUploadHandler() {
        jsonResponse = new HttpRequestResponse();
    }

    /**
     * Handles the upload request by processing the file and testing it..
     *
     * @param ctx the context
     * @param request the full request
     */
    @Override
    protected void channelRead0(ChannelHandlerContext ctx, FullHttpRequest request) {
        QueryStringDecoder decoder = new QueryStringDecoder(request.uri());
        String modelName = NettyUtils.getParameter(decoder, "modelName", null);
        Map<String, String> names = new ConcurrentHashMap<>();
        CompletableFuture.supplyAsync(
                        () -> {
                            if (modelName != null) {
                                names.put(modelName, modelName);
                                return names;
                            } else {
                                throw new BadRequestException("modelName is mandatory.");
                            }
                        })
                .exceptionally((ex) -> Collections.emptyMap())
                .thenAccept(nameResponse -> jsonResponse.sendAsJson(ctx, request, nameResponse));
    }

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) {
        FullHttpRequest request = (FullHttpRequest) msg;

        String uri = request.uri();
        return uri.startsWith("/uploading/models?");
    }
}

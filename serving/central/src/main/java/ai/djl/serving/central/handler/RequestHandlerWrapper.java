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

import ai.djl.serving.central.responseencoder.JsonResponse;
import ai.djl.serving.plugins.RequestHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.util.concurrent.CompletableFuture;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * wraps a Request-Handler into a SimpleChannelInboundHandler to use central as a stand-alone
 * component without serving.
 *
 * @author erik.bamberg@web.de
 */
public class RequestHandlerWrapper extends SimpleChannelInboundHandler<FullHttpRequest> {

    private static final Logger logger = LoggerFactory.getLogger(RequestHandlerWrapper.class);

    private RequestHandler<?> wrappedRequestHandler;
    private JsonResponse jsonResponse;

    /**
     * wrap handler.
     *
     * @param wrappedRequestHandler the requesthandler to wrap.
     */
    public RequestHandlerWrapper(RequestHandler<?> wrappedRequestHandler) {
        this.wrappedRequestHandler = wrappedRequestHandler;
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
        return wrappedRequestHandler.acceptInboundMessage(msg);
    }

    /**
     * handle requests.
     *
     * @param ctx the context
     * @param request the full request
     */
    @SuppressWarnings("unchecked")
    @Override
    protected void channelRead0(ChannelHandlerContext ctx, FullHttpRequest request) {
        try {
            if (!request.decoderResult().isSuccess()) {
                throw new ai.djl.serving.http.BadRequestException("Invalid HTTP message.");
            }

            QueryStringDecoder decoder = new QueryStringDecoder(request.uri());

            String path = decoder.path();

            String[] segments = path.split("/");
            Object response = wrappedRequestHandler.handleRequest(ctx, request, decoder, segments);
            if (response instanceof CompletableFuture) {
                ((CompletableFuture<Object>) response)
                        .thenAccept(resultOrError -> jsonResponse.send(ctx, request, resultOrError))
                        .exceptionally(
                                (ex) -> {
                                    logger.error("error handling request", ex);
                                    jsonResponse.send(ctx, request, ex);
                                    return null;
                                });
            } else if (response != null) {
                jsonResponse.send(ctx, request, response);
            }
        } catch (Throwable t) {
            logger.error(t.getMessage(), t);
            jsonResponse.send(ctx, request, t);
        }
    }
}

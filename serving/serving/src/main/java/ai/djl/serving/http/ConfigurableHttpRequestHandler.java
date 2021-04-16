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
package ai.djl.serving.http;

import ai.djl.ModelException;
import ai.djl.serving.plugins.FolderScanPluginManager;
import ai.djl.serving.plugins.RequestHandler;
import ai.djl.serving.util.NettyUtils;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * HttpRequestHandler that tries to process a http-request using the configured RequestHandlers.
 *
 * <p>RequestHandlers are configured by the PluginManager.
 *
 * @author erik.bamberg@web.de
 */
public class ConfigurableHttpRequestHandler extends HttpRequestHandler {

    private static final Logger logger =
            LoggerFactory.getLogger(ConfigurableHttpRequestHandler.class);

    private FolderScanPluginManager pluginManager;

    /**
     * constructing a ConfigurableHttpRequestHandler.
     *
     * @param pluginManager a pluginManager instance used to search for available plug-ins to
     *     process a request.
     */
    public ConfigurableHttpRequestHandler(FolderScanPluginManager pluginManager) {
        this.pluginManager = pluginManager;
    }

    /** {@inheritDoc} */
    @SuppressWarnings("unchecked")
    @Override
    protected void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException {
        RequestHandler<?> requestHandler =
                findRequestHandler(req)
                        .orElseThrow(
                                () -> new BadRequestException("request handler no longer valid"));
        logger.debug(
                "request handler {} processes request ", requestHandler.getClass().getSimpleName());
        try {
            Object result = requestHandler.handleRequest(ctx, req, decoder, segments);
            if (result != null) {
                if (result instanceof CompletableFuture) {
                    ((CompletableFuture<Object>) result)
                            .handle(
                                    (response, error) -> {
                                        if (error != null) {
                                            NettyUtils.sendError(ctx, error);
                                        } else {
                                            NettyUtils.sendJsonResponse(ctx, response);
                                        }
                                        return response;
                                    });
                } else {
                    NettyUtils.sendJsonResponse(ctx, result);
                }
            }
        } catch (Exception ex) {
            NettyUtils.sendError(ctx, ex);
        }
    }

    /**
     * findRequestHandler.
     *
     * @param req the full Http Request.
     * @return an optional RequestHandler.
     */
    @SuppressWarnings("rawtypes")
    private Optional<RequestHandler> findRequestHandler(FullHttpRequest req) {
        return pluginManager
                .findImplementations(RequestHandler.class)
                .stream()
                .filter(h -> h.acceptInboundMessage(req))
                .findFirst();
    }

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) throws Exception {
        if (msg instanceof FullHttpRequest) {
            return findRequestHandler((FullHttpRequest) msg).isPresent();
        } else {
            return false;
        }
    }
}

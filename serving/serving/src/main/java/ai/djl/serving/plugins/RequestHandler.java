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
package ai.djl.serving.plugins;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;

/**
 * Interface to be implemented by HtttpRequestHandler.
 *
 * <p>Classes implementing this interface and populated as service using the SPI service-manifest
 * are pickup by the serving plugin architectur and automatically registered as RequestHandler for
 * HTTP Requests.
 *
 * @author erik.bamberg@web.de
 */
public interface RequestHandler<T> {

    /**
     * Returns true if this handler can handle the incoming HTTP request.
     *
     * <p>The interface following the chain of responsibility pattern.
     *
     * @param msg the incoming HTTP message
     * @return true if this handler can handle the incoming HTTP request. false otherwise
     */
    boolean acceptInboundMessage(Object msg);

    /**
     * The main method which handles request.
     *
     * <p>This method is called by the framework if {@code acceptInboundMessage} indicates that this
     * handler can handle the request.
     *
     * @param ctx the handler context.
     * @param req the full HttpRequest object.
     * @param decoder a query string decoder helps to parse the url query string.
     * @param segments array of splitted segments of the path.
     * @return a response or null. The response is returned to the client converting it to the
     *     requested format by the server.
     */
    T handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments);
}

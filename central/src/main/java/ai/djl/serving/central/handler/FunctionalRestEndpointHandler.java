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
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.serving.central.responseencoder.HttpRequestResponse;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import java.io.IOException;
import java.net.URI;
import java.util.Collections;
import java.util.concurrent.CompletableFuture;
import java.util.function.BiConsumer;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.regex.Pattern;

/**
 * 
 * A Generic Http Request Handler which passes the work to a {@code java.function.Supplier} and response with a json object.
 * 
 * @author erik.bamberg@web.de
 *
 */
public class FunctionalRestEndpointHandler extends SimpleChannelInboundHandler<FullHttpRequest> {

   
    private BiConsumer<ChannelHandlerContext,FullHttpRequest> consumer;
    private Pattern pattern;
    private HttpMethod method;
    
    /**
     * Constructs a endpoint handler with a supplier and a url-regex-pattern.
     * @param supplier this function is called.
     * @param pattern this handler is used when the requested url matches this regex. 
     * 
     */
    public FunctionalRestEndpointHandler(BiConsumer<ChannelHandlerContext,FullHttpRequest> consumer, Pattern pattern, HttpMethod method) {
	 this.consumer=consumer;
	 this.pattern=pattern;
	 this.method=method;
    }
    
    /**
     * handle get Model meta data requests.
     *
     * @param ctx the context
     * @param request the full request
     */
    @Override
    protected void channelRead0(ChannelHandlerContext ctx, FullHttpRequest request) {	
        consumer.accept(ctx, request);
    }

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) throws Exception {
        if (super.acceptInboundMessage(msg)) {
            FullHttpRequest req = (FullHttpRequest) msg;
            URI uri=new URI(req.uri());
            return this.method.equals(req.method()) && pattern.matcher(uri.getPath()).matches();
        }
        return false;
    }

}
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
package ai.djl.serving.central.client;

import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpContent;
import io.netty.handler.codec.http.HttpObject;
import io.netty.handler.codec.http.HttpResponse;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.LastHttpContent;
import io.netty.util.CharsetUtil;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A client handler that expects objects in JSON format.
 * 
 * @author erik.bamberg@web.de
 *
 */
public class CallbackClientHandler extends SimpleChannelInboundHandler<HttpObject> {

    private static final Logger logger = LoggerFactory.getLogger(CallbackClientHandler.class);

    private Consumer<FullHttpResponse> callback;
    
    /**
     * Constructs a JSON client Handler.
     */
    CallbackClientHandler(Consumer<FullHttpResponse> callback) {
	this.callback=callback;
    }

    @Override
    public void channelRead0(ChannelHandlerContext ctx, HttpObject msg) {
	if (msg instanceof FullHttpResponse) {
	    FullHttpResponse response = (FullHttpResponse) msg;
	    logger.debug(response.toString());
	    callback.accept(response);
	}
	

    }

    /**
     * exception handling.
     * 
     * @param ctx   the context to send data to the server
     * @param cause the exception
     */
    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
	// Close the connection when an exception is raised.
	logger.error(cause.getMessage(), cause);
	ctx.close();
    }

}

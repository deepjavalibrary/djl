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
package ai.djl.serving.central;

import ai.djl.serving.central.handler.ModelListMetaDataHandler;
import ai.djl.serving.central.handler.ModelMetaDataHandler;
import ai.djl.serving.central.handler.RequestHandlerWrapper;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.socket.SocketChannel;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpServerCodec;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.stream.ChunkedWriteHandler;

/**
 * initializer to setup netty instance.
 *
 * @author erik.bamberg@web.de
 */
public class HttpStaticFileServerInitializer extends ChannelInitializer<SocketChannel> {

    private SslContext sslCtx;

    /**
     * constructs the initializer.
     *
     * @param sslCtx the ssl context to use.
     */
    public HttpStaticFileServerInitializer(SslContext sslCtx) {
        this.sslCtx = sslCtx;
    }

    /**
     * init and configure the channel.
     *
     * @param ch the socketchannel
     */
    @Override
    public void initChannel(SocketChannel ch) {
        ChannelPipeline pipeline = ch.pipeline();
        if (sslCtx != null) {
            pipeline.addLast(sslCtx.newHandler(ch.alloc()));
        }
        pipeline.addLast(new HttpServerCodec());
        pipeline.addLast(new HttpObjectAggregator(65536));
        pipeline.addLast(new ChunkedWriteHandler());
        pipeline.addLast(new RequestHandlerWrapper(new ModelListMetaDataHandler()));
        pipeline.addLast(new RequestHandlerWrapper(new ModelMetaDataHandler()));
    }
}

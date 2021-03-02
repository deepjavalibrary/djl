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
package ai.djl.serving.central.responseencoder;

import ai.djl.modality.Classifications;
import ai.djl.modality.Classifications.ClassificationsSerializer;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.Metadata;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializer;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.util.CharsetUtil;
import java.lang.reflect.Modifier;

/**
 * serialize to json and send the response to the client.
 *
 * @author erik.bamberg@web.de
 */
public class HttpRequestResponse {

    private static final Gson GSON_WITH_TRANSIENT_FIELDS =
            new GsonBuilder()
                    .setDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
                    .setPrettyPrinting()
                    .excludeFieldsWithModifiers(Modifier.STATIC)
                    .registerTypeAdapter(Classifications.class, new ClassificationsSerializer())
                    .registerTypeAdapter(DetectedObjects.class, new ClassificationsSerializer())
                    .registerTypeAdapter(Metadata.class, new MetaDataSerializer())
                    .registerTypeAdapter(
                            Double.class,
                            (JsonSerializer<Double>)
                                    (src, t, ctx) -> {
                                        long v = src.longValue();
                                        if (src.equals(Double.valueOf(String.valueOf(v)))) {
                                            return new JsonPrimitive(v);
                                        }
                                        return new JsonPrimitive(src);
                                    })
                    .create();

    /**
     * send a response to the client.
     *
     * @param ctx channel context
     * @param request full request
     * @param entity the response
     */
    public void sendAsJson(ChannelHandlerContext ctx, FullHttpRequest request, Object entity) {

        String serialized = GSON_WITH_TRANSIENT_FIELDS.toJson(entity);
        ByteBuf buffer = ctx.alloc().buffer(serialized.length());
        buffer.writeCharSequence(serialized, CharsetUtil.UTF_8);

        FullHttpResponse response =
                new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.OK, buffer);
        response.headers().set(HttpHeaderNames.CONTENT_TYPE, "application/json; charset=UTF-8");
        boolean keepAlive = HttpUtil.isKeepAlive(request);
        this.sendAndCleanupConnection(ctx, response, keepAlive);
    }

    /**
     * send content of a ByteBuffer as
     *  response to the client.
     *
     * @param ctx channel context
     * @param request full request
     * @param entity the response
     */
    public void sendByteBuffer(ChannelHandlerContext ctx, ByteBuf buffer) {

        FullHttpResponse response =
                new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.OK, buffer);
        response.headers().set(HttpHeaderNames.CONTENT_TYPE, "application/json; charset=UTF-8");
        this.sendAndCleanupConnection(ctx, response, false);
    }

    /**
     * If Keep-Alive is disabled, attaches "Connection: close" header to the response and closes the
     * connection after the response being sent.
     *
     * @param ctx context
     * @param request full request
     * @param response full response
     */
    private void sendAndCleanupConnection(
            ChannelHandlerContext ctx, FullHttpResponse response, boolean keepAlive ) {
        HttpUtil.setContentLength(response, response.content().readableBytes());
        if (!keepAlive) {
            // We're going to close the connection as soon as the response is sent,
            // so we should also make it clear for the client.
            response.headers().set(HttpHeaderNames.CONNECTION, HttpHeaderValues.CLOSE);
        }

        ChannelFuture flushPromise = ctx.writeAndFlush(response);

        if (!keepAlive) {
            // Close the connection as soon as the response is sent.
            flushPromise.addListener(ChannelFutureListener.CLOSE);
        }
    }
}

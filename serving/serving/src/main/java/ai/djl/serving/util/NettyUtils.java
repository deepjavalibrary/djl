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
package ai.djl.serving.util;

import ai.djl.ModelException;
import ai.djl.modality.Input;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.serving.http.BadRequestException;
import ai.djl.serving.http.ErrorResponse;
import ai.djl.serving.http.MethodNotAllowedException;
import ai.djl.serving.http.ResourceNotFoundException;
import ai.djl.serving.http.ServiceUnavailableException;
import ai.djl.serving.http.Session;
import ai.djl.util.JsonSerializable;
import ai.djl.util.JsonUtils;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpHeaders;
import io.netty.handler.codec.http.HttpRequest;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.handler.codec.http.multipart.Attribute;
import io.netty.handler.codec.http.multipart.FileUpload;
import io.netty.handler.codec.http.multipart.InterfaceHttpData;
import io.netty.util.AttributeKey;
import io.netty.util.CharsetUtil;
import java.io.IOException;
import java.net.SocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A utility class that handling Netty request and response. */
public final class NettyUtils {

    private static final Logger logger = LoggerFactory.getLogger("ACCESS_LOG");

    private static final String REQUEST_ID = "x-request-id";
    private static final AttributeKey<Session> SESSION_KEY = AttributeKey.valueOf("session");

    private NettyUtils() {}

    /**
     * Updates session when a HTTP request is received.
     *
     * @param channel the connection channel
     * @param request the HTTP request
     */
    public static void requestReceived(Channel channel, HttpRequest request) {
        Session session = channel.attr(SESSION_KEY).get();
        assert session == null;

        SocketAddress address = channel.remoteAddress();
        String remoteIp;
        if (address == null) {
            // This is can be null on UDS, or on certain case in Windows
            remoteIp = "0.0.0.0";
        } else {
            remoteIp = address.toString();
        }
        channel.attr(SESSION_KEY).set(new Session(remoteIp, request));
    }

    /**
     * Returns the request ID for the specified channel.
     *
     * @param channel the connection channel
     * @return the request ID for the specified channel
     */
    public static String getRequestId(Channel channel) {
        Session accessLog = channel.attr(SESSION_KEY).get();
        if (accessLog != null) {
            return accessLog.getRequestId();
        }
        return null;
    }

    /**
     * Sends the json object to client.
     *
     * @param ctx the connection context
     * @param obj the object to be sent
     */
    public static void sendJsonResponse(ChannelHandlerContext ctx, Object obj) {
        sendJsonResponse(ctx, obj, HttpResponseStatus.OK);
    }

    /**
     * Sends the json string to client with specified status.
     *
     * @param ctx the connection context
     * @param obj the object to be sent
     * @param status the HTTP status
     */
    public static void sendJsonResponse(
            ChannelHandlerContext ctx, Object obj, HttpResponseStatus status) {
        String content;
        if (obj instanceof JsonSerializable) {
            content = ((JsonSerializable) obj).toJson();
        } else {
            content = JsonUtils.GSON_PRETTY.toJson(obj);
        }
        sendJsonResponse(ctx, content, status);
    }

    /**
     * Sends the json string to client.
     *
     * @param ctx the connection context
     * @param json the json string
     */
    public static void sendJsonResponse(ChannelHandlerContext ctx, String json) {
        sendJsonResponse(ctx, json, HttpResponseStatus.OK);
    }

    /**
     * Sends the json object to client with specified status.
     *
     * @param ctx the connection context
     * @param json the json object
     * @param status the HTTP status
     */
    public static void sendJsonResponse(
            ChannelHandlerContext ctx, String json, HttpResponseStatus status) {
        FullHttpResponse resp = new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, status, false);
        resp.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        ByteBuf content = resp.content();
        content.writeCharSequence(json, CharsetUtil.UTF_8);
        content.writeByte('\n');
        sendHttpResponse(ctx, resp, true);
    }

    /**
     * Sends error to client with exception.
     *
     * @param ctx the connection context
     * @param t the exception to be send
     */
    public static void sendError(ChannelHandlerContext ctx, Throwable t) {
        if (t instanceof ResourceNotFoundException || t instanceof ModelNotFoundException) {
            logger.trace("", t);
            NettyUtils.sendError(ctx, HttpResponseStatus.NOT_FOUND, t);
        } else if (t instanceof BadRequestException || t instanceof ModelException) {
            logger.trace("", t);
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST, t);
        } else if (t instanceof MethodNotAllowedException) {
            logger.trace("", t);
            NettyUtils.sendError(ctx, HttpResponseStatus.METHOD_NOT_ALLOWED, t);
        } else if (t instanceof ServiceUnavailableException) {
            logger.trace("", t);
            NettyUtils.sendError(ctx, HttpResponseStatus.SERVICE_UNAVAILABLE, t);
        } else {
            logger.error("", t);
            NettyUtils.sendError(ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR, t);
        }
    }

    /**
     * Sends error to client with HTTP status and exception.
     *
     * @param ctx the connection context
     * @param status the HTTP status
     * @param t the exception to be send
     */
    public static void sendError(
            ChannelHandlerContext ctx, HttpResponseStatus status, Throwable t) {
        ErrorResponse error =
                new ErrorResponse(status.code(), t.getClass().getSimpleName(), t.getMessage());
        sendJsonResponse(ctx, error, status);
    }

    /**
     * Send HTTP response to client.
     *
     * @param ctx ChannelHandlerContext
     * @param resp HttpResponse to send
     * @param keepAlive if keep the connection
     */
    public static void sendHttpResponse(
            ChannelHandlerContext ctx, FullHttpResponse resp, boolean keepAlive) {
        // Send the response and close the connection if necessary.
        Channel channel = ctx.channel();
        Session session = channel.attr(SESSION_KEY).getAndSet(null);
        HttpHeaders headers = resp.headers();

        ConfigManager configManager = ConfigManager.getInstance();
        int code = resp.status().code();
        if (session != null) {
            // session might be recycled if channel is closed already.
            session.setCode(code);
            headers.set(REQUEST_ID, session.getRequestId());
            logger.info(session.toString());
        } else {
            logger.info("HTTP " + code);
        }

        String allowedOrigin = configManager.getCorsAllowedOrigin();
        String allowedMethods = configManager.getCorsAllowedMethods();
        String allowedHeaders = configManager.getCorsAllowedHeaders();

        if (allowedOrigin != null
                && !allowedOrigin.isEmpty()
                && !headers.contains(HttpHeaderNames.ACCESS_CONTROL_ALLOW_ORIGIN)) {
            headers.set(HttpHeaderNames.ACCESS_CONTROL_ALLOW_ORIGIN, allowedOrigin);
        }
        if (allowedMethods != null
                && !allowedMethods.isEmpty()
                && !headers.contains(HttpHeaderNames.ACCESS_CONTROL_ALLOW_METHODS)) {
            headers.set(HttpHeaderNames.ACCESS_CONTROL_ALLOW_METHODS, allowedMethods);
        }
        if (allowedHeaders != null
                && !allowedHeaders.isEmpty()
                && !headers.contains(HttpHeaderNames.ACCESS_CONTROL_ALLOW_HEADERS)) {
            headers.set(HttpHeaderNames.ACCESS_CONTROL_ALLOW_HEADERS, allowedHeaders);
        }

        // Add cache-control headers to avoid browser cache response
        headers.set("Pragma", "no-cache");
        headers.set("Cache-Control", "no-cache; no-store, must-revalidate, private");
        headers.set("Expires", "Thu, 01 Jan 1970 00:00:00 UTC");

        HttpUtil.setContentLength(resp, resp.content().readableBytes());
        if (!keepAlive || code >= 400) {
            headers.set(HttpHeaderNames.CONNECTION, HttpHeaderValues.CLOSE);
            ChannelFuture f = channel.writeAndFlush(resp);
            f.addListener(ChannelFutureListener.CLOSE);
        } else {
            headers.set(HttpHeaderNames.CONNECTION, HttpHeaderValues.KEEP_ALIVE);
            channel.writeAndFlush(resp);
        }
    }

    /**
     * Closes the specified channel after all queued write requests are flushed.
     *
     * @param ch the channel to be closed
     */
    public static void closeOnFlush(Channel ch) {
        if (ch.isActive()) {
            ch.writeAndFlush(Unpooled.EMPTY_BUFFER).addListener(ChannelFutureListener.CLOSE);
        }
    }

    /**
     * Returns the bytes for the specified {@code ByteBuf}.
     *
     * @param buf the {@code ByteBuf} to read
     * @return the bytes for the specified {@code ByteBuf}
     */
    public static byte[] getBytes(ByteBuf buf) {
        if (buf.hasArray()) {
            return buf.array();
        }

        byte[] ret = new byte[buf.readableBytes()];
        int readerIndex = buf.readerIndex();
        buf.getBytes(readerIndex, ret);
        return ret;
    }

    /**
     * Reads the parameter's value for the key from the uri.
     *
     * @param decoder the {@code QueryStringDecoder} parsed from uri
     * @param key the parameter key
     * @param def the default value
     * @return the parameter's value
     */
    public static String getParameter(QueryStringDecoder decoder, String key, String def) {
        List<String> param = decoder.parameters().get(key);
        if (param != null && !param.isEmpty()) {
            return param.get(0);
        }
        return def;
    }

    /**
     * Read the parameter's integer value for the key from the uri.
     *
     * @param decoder the {@code QueryStringDecoder} parsed from uri
     * @param key the parameter key
     * @param def the default value
     * @return the parameter's integer value
     * @throws NumberFormatException exception is thrown when the parameter-value is not numeric.
     */
    public static int getIntParameter(QueryStringDecoder decoder, String key, int def) {
        String value = getParameter(decoder, key, null);
        if (value == null || value.isEmpty()) {
            return def;
        }
        return Integer.parseInt(value);
    }

    /**
     * Parses form data and added to the {@link Input} object.
     *
     * @param data the form data
     * @param input the {@link Input} object to be added to
     */
    public static void addFormData(InterfaceHttpData data, Input input) {
        if (data == null) {
            return;
        }
        try {
            String name = data.getName();
            switch (data.getHttpDataType()) {
                case Attribute:
                    Attribute attribute = (Attribute) data;
                    input.addData(name, attribute.getValue().getBytes(StandardCharsets.UTF_8));
                    break;
                case FileUpload:
                    FileUpload fileUpload = (FileUpload) data;
                    input.addData(name, getBytes(fileUpload.getByteBuf()));
                    break;
                default:
                    throw new IllegalArgumentException(
                            "Except form field, but got " + data.getHttpDataType());
            }
        } catch (IOException e) {
            throw new AssertionError(e);
        }
    }
}

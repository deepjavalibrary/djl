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
package ai.djl.serving.plugins.staticfile;

import ai.djl.repository.FilenameUtils;
import ai.djl.serving.http.ResourceNotFoundException;
import ai.djl.serving.plugins.RequestHandler;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.DefaultHttpResponse;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpChunkedInput;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponse;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.handler.stream.ChunkedNioStream;
import io.netty.util.AsciiString;
import java.io.IOException;
import java.io.InputStream;
import java.net.JarURLConnection;
import java.net.URL;
import java.net.URLConnection;
import java.nio.channels.ReadableByteChannel;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.GregorianCalendar;
import java.util.Locale;
import java.util.TimeZone;
import java.util.jar.JarEntry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * handler to publish file resources classpath.
 *
 * <p>this handler exposes every file which is in /static/-folder in the classpath.
 *
 * @author erik.bamberg@web.de
 */
public class HttpStaticClasspathResourceHandler implements RequestHandler<Void> {

    private static final Logger logger =
            LoggerFactory.getLogger(HttpStaticClasspathResourceHandler.class);

    private static final String HTTP_DATE_FORMAT = "EEE, dd MMM yyyy HH:mm:ss zzz";
    private static final String HTTP_DATE_GMT_TIMEZONE = "GMT";
    private static final int HTTP_CACHE_SECONDS = 60;
    private static final int HTTP_CHUNK_SIZE = 8192;

    private static final String RESOURCE_FOLDER = "/static";

    /**
     * chain of responsibility. accept-method.
     *
     * @param msg the request message to handle.
     * @return true/false if this handler can handler this kind of request.
     */
    @Override
    public boolean acceptInboundMessage(Object msg) {
        if (!(msg instanceof FullHttpRequest)) {
            return false;
        }

        FullHttpRequest request = (FullHttpRequest) msg;
        if (!HttpMethod.GET.equals(request.method())) {
            return false;
        } else {
            String uri = request.uri();
            return ("/".equals(uri) || this.getClass().getResource(RESOURCE_FOLDER + uri) != null);
        }
    }

    /** {@inheritDoc} */
    @Override
    public Void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest request,
            QueryStringDecoder decoder,
            String[] segments) {
        logger.debug("processing static resource request");
        String uri = request.uri();

        uri = mapToPathInResourceFolder(uri);

        try {
            URL resource = getClass().getResource(uri);
            if (resource != null) {
                ResourceInfo resourceInfo = getResourceInfo(resource);
                logger.debug(resourceInfo.toString());

                sendHeader(ctx, request, resourceInfo);

                try (InputStream resourceInputStream = resource.openStream()) {
                    // Create a NIO ReadableByteChannel from the stream
                    ReadableByteChannel channel =
                            java.nio.channels.Channels.newChannel(resourceInputStream);

                    ChannelFuture sendFileFuture =
                            ctx.writeAndFlush(
                                    new HttpChunkedInput(
                                            new ChunkedNioStream(channel, HTTP_CHUNK_SIZE)),
                                    ctx.newProgressivePromise());
                    if (!HttpUtil.isKeepAlive(request)) {
                        // Close the connection when the whole content is written out.
                        sendFileFuture.addListener(ChannelFutureListener.CLOSE);
                    }
                }

            } else {
                throw new ResourceNotFoundException();
            }

        } catch (IOException ex) {
            logger.error("error reading requested resource", ex);
            throw new ResourceNotFoundException(ex);
        }
        return null;
    }

    private void sendHeader(
            ChannelHandlerContext ctx, FullHttpRequest request, ResourceInfo resourceInfo) {
        // send header
        boolean keepAlive = HttpUtil.isKeepAlive(request);
        HttpResponse response =
                new DefaultHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.OK);
        HttpUtil.setContentLength(response, resourceInfo.getSize());
        setContentTypeHeader(response, resourceInfo);
        setDateAndCacheHeaders(response, resourceInfo);

        if (!keepAlive) {
            response.headers().set(HttpHeaderNames.CONNECTION, HttpHeaderValues.CLOSE);
        } else if (request.protocolVersion().equals(HttpVersion.HTTP_1_0)) {
            response.headers().set(HttpHeaderNames.CONNECTION, HttpHeaderValues.KEEP_ALIVE);
        }

        // Write the initial line and the header.
        ctx.write(response);
    }

    private String mapToPathInResourceFolder(String uri) {
        StringBuilder newURI = new StringBuilder(RESOURCE_FOLDER);
        if ("/".equals(uri)) {
            newURI.append("/index.html");
        } else {
            newURI.append(uri);
        }
        uri = newURI.toString();
        return uri;
    }

    /**
     * Retrieve the last modified date of the connection.
     *
     * @param resourceURL the url
     * @return resourceInfo which file-information like size and modifiedDate
     * @throws IOException accessing the entry
     */
    private ResourceInfo getResourceInfo(URL resourceURL) throws IOException {
        URLConnection connection;

        connection = resourceURL.openConnection();
        if (connection instanceof JarURLConnection) {
            JarURLConnection jarConnection = ((JarURLConnection) connection);
            JarEntry entry = jarConnection.getJarEntry();
            if (entry != null) {
                return new ResourceInfo(entry.getName(), entry.getTime(), entry.getSize());
            }
            return ResourceInfo.empty();
        } else {
            try {
                return new ResourceInfo(
                        connection.getURL().toString(),
                        connection.getLastModified(),
                        connection.getContentLengthLong());
            } finally {
                connection.getInputStream().close();
            }
        }
    }

    /**
     * Sets the content type header for the HTTP Response.
     *
     * @param response HTTP response
     * @param resourceInfo resourceInfo to extract content type
     */
    private static void setContentTypeHeader(HttpResponse response, ResourceInfo resourceInfo) {
        String ext = FilenameUtils.getFileExtension(resourceInfo.getName());
        AsciiString contentType = MimeUtils.getContentType(ext);
        response.headers().set(HttpHeaderNames.CONTENT_TYPE, contentType);
    }

    /**
     * Sets the Date and Cache headers for the HTTP Response.
     *
     * @param response HTTP response
     * @param resourceInfo to get the last modified time of the file
     */
    private static void setDateAndCacheHeaders(HttpResponse response, ResourceInfo resourceInfo) {
        SimpleDateFormat dateFormatter = new SimpleDateFormat(HTTP_DATE_FORMAT, Locale.US);
        dateFormatter.setTimeZone(TimeZone.getTimeZone(HTTP_DATE_GMT_TIMEZONE));

        // Date header
        Calendar time = new GregorianCalendar();
        response.headers().set(HttpHeaderNames.DATE, dateFormatter.format(time.getTime()));

        // Add cache headers
        time.add(Calendar.SECOND, HTTP_CACHE_SECONDS);
        response.headers().set(HttpHeaderNames.EXPIRES, dateFormatter.format(time.getTime()));
        response.headers()
                .set(HttpHeaderNames.CACHE_CONTROL, "private, max-age=" + HTTP_CACHE_SECONDS);
        response.headers()
                .set(
                        HttpHeaderNames.LAST_MODIFIED,
                        dateFormatter.format(new Date(resourceInfo.getLastModified())));
    }

    static final class ResourceInfo {
        private String name;
        private long lastModified;
        private long size;

        public ResourceInfo(String name, long lastModified, long size) {
            super();
            this.name = name;
            this.lastModified = lastModified;
            this.size = size;
        }

        public static ResourceInfo empty() {
            return new ResourceInfo("not found", 0L, 0L);
        }

        public String getName() {
            return name;
        }

        public long getLastModified() {
            return lastModified;
        }

        public long getSize() {
            return size;
        }

        @Override
        public String toString() {
            return "resource " + name + " [ size:" + size + " lastModified:" + lastModified + " ]";
        }
    }
}

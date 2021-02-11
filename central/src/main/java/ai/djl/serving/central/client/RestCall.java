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

import io.netty.bootstrap.Bootstrap;
import io.netty.buffer.Unpooled;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelOption;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.handler.codec.http.DefaultFullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpClientCodec;
import io.netty.handler.codec.http.HttpContent;
import io.netty.handler.codec.http.HttpContentDecompressor;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpRequest;
import io.netty.handler.codec.http.HttpResponse;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.cookie.ClientCookieEncoder;
import io.netty.handler.codec.http.cookie.DefaultCookie;
import io.netty.handler.codec.json.JsonObjectDecoder;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.util.InsecureTrustManagerFactory;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;
import javax.net.ssl.SSLException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A reactive non-blocking rest client.
 * 
 * @author erik.bamberg@web.de
 *
 */
public class RestCall implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(RestCall.class);
    static final boolean isSSLEnabled = System.getProperty("ssl") != null;
    static final int SIZE = Integer.parseInt(System.getProperty("size", "256"));

    private EventLoopGroup workerGroup;
    private Bootstrap bootStrap;
    
   

    /**
     * Creates a new web client.
     */
    public RestCall(Consumer<FullHttpResponse> callback) {
	
	workerGroup = new NioEventLoopGroup();

	SslContext sslCtx = buildSSLContext();

	bootStrap = new Bootstrap();
	bootStrap.group(workerGroup).channel(NioSocketChannel.class).option(ChannelOption.TCP_NODELAY, true)
		.handler(new ChannelInitializer<SocketChannel>() {

		    @Override
		    public void initChannel(SocketChannel ch) throws Exception {
			ChannelPipeline p = ch.pipeline();
			if (sslCtx != null) {
			    p.addLast(sslCtx.newHandler(ch.alloc()));
			}
			p.addLast(new HttpClientCodec());
			p.addLast(new HttpContentDecompressor());
			p.addLast(new HttpObjectAggregator(Integer.MAX_VALUE));
			p.addLast(new CallbackClientHandler(callback));
		    }
		});

    }

    /**
     * @return
     */
    private SslContext buildSSLContext() {
	SslContext sslCtx;
	try {
	    if (isSSLEnabled) {
		sslCtx = SslContextBuilder.forClient().trustManager(InsecureTrustManagerFactory.INSTANCE).build();
	    } else {
		sslCtx = null;
	    }
	} catch (SSLException e) {
	    logger.error("unable to build ssl context", e);
	    sslCtx=null;
	}
	return sslCtx;
    }

    /**
     * sends a GET returning the response.
     * @param url the url to call
     */
    public void send(String url) {
	try {
	    URI uri = new URI(url);
	    String host = parseHost(uri);
	    int port = parsePort(uri);
	    
	    HttpRequest request = prepareHttpRequest(uri,HttpMethod.GET);
	    
	    ChannelFuture f = bootStrap.connect(host, port).addListener( (future)-> {
		if (future.isSuccess()) {
		    ((ChannelFuture) future).channel()
		    	.writeAndFlush(request)
		    	.addListener( (fut2)-> {
		    	    			  ((ChannelFuture) fut2).channel().closeFuture();
		    	    		       });
		} else {
		    logger.error("cannot establish connection to server",future.cause());
		}
	    });	    
	    
	} catch (URISyntaxException e) {
	    logger.error(e.getMessage(), e);
	} 
    }

    /**
     * prepare a defaultHttpRequest Object for this request
     * 
     * @param uri
     * @param host
     * @return
     */
    private HttpRequest prepareHttpRequest(URI uri,HttpMethod method) {
	// Prepare the HTTP request.
	HttpRequest request = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, method, uri.getRawPath(),
		Unpooled.EMPTY_BUFFER);
	request.headers().set(HttpHeaderNames.HOST, parseHost(uri));
	request.headers().set(HttpHeaderNames.CONNECTION, HttpHeaderValues.CLOSE);
	request.headers().set(HttpHeaderNames.ACCEPT_ENCODING, HttpHeaderValues.GZIP);
	return request;
    }


    private String parseScheme(URI uri) {
	String scheme = uri.getScheme() == null ? "http" : uri.getScheme();
	if (!"http".equalsIgnoreCase(scheme) && !"https".equalsIgnoreCase(scheme)) {
	    throw new IllegalArgumentException("Only HTTP(S) is supported.");
	}
	return scheme;
    }

    private String parseHost(URI uri) {
	String host = uri.getHost() == null ? "127.0.0.1" : uri.getHost();
	return host;
    }

    private int parsePort(URI uri) {
	String scheme = parseScheme(uri);
	int port = uri.getPort();
	if (port == -1) {
	    if ("http".equalsIgnoreCase(scheme)) {
		port = 80;
	    } else if ("https".equalsIgnoreCase(scheme)) {
		port = 443;
	    }
	}
	return port;

    }

    @Override
    public void close() throws Exception {
	if (workerGroup!=null) {
	    workerGroup.shutdownGracefully();
	}
    }

    private class Builder {
	
    }
    
}

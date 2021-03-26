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

import io.netty.bootstrap.ServerBootstrap;
import io.netty.channel.Channel;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.handler.logging.LogLevel;
import io.netty.handler.logging.LoggingHandler;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.SslProvider;
import io.netty.handler.ssl.util.SelfSignedCertificate;
import java.security.cert.CertificateException;
import javax.net.ssl.SSLException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * a webserver to browse models in ModelZoo.
 *
 * @author erik.bamberg@web.de
 */
public class ModelZooRepositoryServer {

    private static final Logger logger = LoggerFactory.getLogger(ModelZooRepositoryServer.class);

    static final boolean SSL = System.getProperty("ssl") != null;
    static final int PORT = Integer.parseInt(System.getProperty("port", SSL ? "8443" : "8080"));

    /**
     * starts ModelZoo server.
     *
     * @param args program arguments
     */
    public static void main(String[] args) {
        try {
            new ModelZooRepositoryServer().startup();
        } catch (Exception e) {
            logger.error(e.getMessage(), e);
        }
    }

    /**
     * starts the server instance.
     *
     * @throws InterruptedException thread got interrupted
     * @throws SSLException error with ssl setup
     * @throws CertificateException error handling certificates
     */
    public void startup() throws InterruptedException, SSLException, CertificateException {
        // Configure SSL.
        SslContext sslCtx;
        if (SSL) {
            SelfSignedCertificate ssc = new SelfSignedCertificate();
            sslCtx =
                    SslContextBuilder.forServer(ssc.certificate(), ssc.privateKey())
                            .sslProvider(SslProvider.JDK)
                            .build();
        } else {
            sslCtx = null;
        }

        EventLoopGroup bossGroup = new NioEventLoopGroup(1);
        EventLoopGroup workerGroup = new NioEventLoopGroup();
        try {
            ServerBootstrap b = new ServerBootstrap();
            b.group(bossGroup, workerGroup)
                    .channel(NioServerSocketChannel.class)
                    .handler(new LoggingHandler(LogLevel.INFO))
                    .childHandler(new HttpStaticFileServerInitializer(sslCtx));

            Channel ch = b.bind(PORT).sync().channel();

            logger.info(
                    "Open your web browser and navigate to "
                            + (SSL ? "https" : "http")
                            + "://127.0.0.1:"
                            + PORT
                            + '/');

            ch.closeFuture().sync();
        } finally {
            bossGroup.shutdownGracefully();
            workerGroup.shutdownGracefully();
        }
    }
}

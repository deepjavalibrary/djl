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
package ai.djl.serving;

import ai.djl.repository.FilenameUtils;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.Connector;
import ai.djl.serving.util.ServerGroups;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.ModelManager;
import io.netty.bootstrap.ServerBootstrap;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelOption;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.ServerChannel;
import io.netty.handler.ssl.SslContext;
import io.netty.util.internal.logging.InternalLoggerFactory;
import io.netty.util.internal.logging.Slf4JLoggerFactory;
import java.io.IOException;
import java.net.MalformedURLException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.GeneralSecurityException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** The main entry point for model server. */
public class ModelServer {

    private static final Logger logger = LoggerFactory.getLogger(ModelServer.class);

    private ServerGroups serverGroups;
    private List<ChannelFuture> futures = new ArrayList<>(2);
    private AtomicBoolean stopped = new AtomicBoolean(false);

    private ConfigManager configManager;

    /**
     * Creates a new {@code ModelServer} instance.
     *
     * @param configManager the model server configuration
     */
    public ModelServer(ConfigManager configManager) {
        this.configManager = configManager;
        serverGroups = new ServerGroups(configManager);
    }

    /**
     * The entry point for the model server.
     *
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        Options options = Arguments.getOptions();
        try {
            DefaultParser parser = new DefaultParser();
            CommandLine cmd = parser.parse(options, args, null, false);
            Arguments arguments = new Arguments(cmd);
            if (arguments.hasHelp()) {
                printHelp("model-server [OPTIONS]", options);
                return;
            }

            ConfigManager.init(arguments);

            ConfigManager configManager = ConfigManager.getInstance();

            InternalLoggerFactory.setDefaultFactory(Slf4JLoggerFactory.INSTANCE);
            new ModelServer(configManager).startAndWait();
        } catch (IllegalArgumentException e) {
            logger.error("Invalid configuration: " + e.getMessage());
            System.exit(1); // NOPMD
        } catch (ParseException e) {
            printHelp(e.getMessage(), options);
            System.exit(1); // NOPMD
        } catch (Throwable t) {
            logger.error("Unexpected error", t);
            System.exit(1); // NOPMD
        }
    }

    /**
     * Starts the model server and block until server stops.
     *
     * @throws InterruptedException if interrupted
     * @throws IOException if failed to start socket listener
     * @throws GeneralSecurityException if failed to read SSL certificate
     */
    public void startAndWait() throws InterruptedException, IOException, GeneralSecurityException {
        try {
            List<ChannelFuture> channelFutures = start();
            logger.info("Model server started.");
            channelFutures.get(0).sync();
        } finally {
            serverGroups.shutdown(true);
            logger.info("Model server stopped.");
        }
    }

    /**
     * Main Method that prepares the future for the channel and sets up the ServerBootstrap.
     *
     * @return a list of ChannelFuture object
     * @throws InterruptedException if interrupted
     * @throws IOException if failed to start socket listener
     * @throws GeneralSecurityException if failed to read SSL certificate
     */
    public List<ChannelFuture> start()
            throws InterruptedException, IOException, GeneralSecurityException {
        stopped.set(false);

        logger.info(configManager.dumpConfigurations());

        initModelStore();

        Connector inferenceConnector =
                configManager.getConnector(Connector.ConnectorType.INFERENCE);
        Connector managementConnector =
                configManager.getConnector(Connector.ConnectorType.MANAGEMENT);
        inferenceConnector.clean();
        managementConnector.clean();

        EventLoopGroup serverGroup = serverGroups.getServerGroup();
        EventLoopGroup workerGroup = serverGroups.getChildGroup();

        futures.clear();
        if (inferenceConnector.equals(managementConnector)) {
            Connector both = configManager.getConnector(Connector.ConnectorType.BOTH);
            futures.add(initializeServer(both, serverGroup, workerGroup, "Both"));
        } else {
            futures.add(
                    initializeServer(inferenceConnector, serverGroup, workerGroup, "Inference"));
            futures.add(
                    initializeServer(managementConnector, serverGroup, workerGroup, "Management"));
        }

        return futures;
    }

    /**
     * Return if the server is running.
     *
     * @return {@code true} if the server is running
     */
    public boolean isRunning() {
        return !stopped.get();
    }

    /** Stops the model server. */
    public void stop() {
        if (stopped.get()) {
            return;
        }

        stopped.set(true);
        for (ChannelFuture future : futures) {
            future.channel().close();
        }
        serverGroups.shutdown(true);
        serverGroups.reset();
    }

    private ChannelFuture initializeServer(
            Connector connector,
            EventLoopGroup serverGroup,
            EventLoopGroup workerGroup,
            final String purpose)
            throws InterruptedException, IOException, GeneralSecurityException {
        Class<? extends ServerChannel> channelClass = connector.getServerChannel();
        logger.info("Initialize {} server with: {}.", purpose, channelClass.getSimpleName());

        ServerBootstrap b = new ServerBootstrap();
        b.option(ChannelOption.SO_BACKLOG, 1024)
                .channel(channelClass)
                .childOption(ChannelOption.SO_LINGER, 0)
                .childOption(ChannelOption.SO_REUSEADDR, true)
                .childOption(ChannelOption.SO_KEEPALIVE, true);
        b.group(serverGroup, workerGroup);

        SslContext sslCtx = null;
        if (connector.isSsl()) {
            sslCtx = configManager.getSslContext();
        }
        b.childHandler(new ServerInitializer(sslCtx, connector.getType()));

        ChannelFuture future;
        try {
            future = b.bind(connector.getSocketAddress()).sync();
        } catch (Exception e) {
            // https://github.com/netty/netty/issues/2597
            if (e instanceof IOException) {
                throw new IOException("Failed to bind to address: " + connector, e);
            }
            throw e;
        }
        future.addListener(
                (ChannelFutureListener)
                        f -> {
                            if (!f.isSuccess()) {
                                try {
                                    f.get();
                                } catch (InterruptedException | ExecutionException e) {
                                    logger.error("", e);
                                }
                                System.exit(2); // NOPMD
                            }
                            serverGroups.registerChannel(f.channel());
                        });

        future.sync();

        ChannelFuture f = future.channel().closeFuture();
        f.addListener(
                (ChannelFutureListener)
                        listener -> logger.info("{} model server stopped.", connector.getType()));

        logger.info("{} API bind to: {}", purpose, connector);
        return f;
    }

    private void initModelStore() throws IOException {
        ModelManager.init(configManager);
        Set<String> startupModels = ModelManager.getInstance().getStartupModels();

        String loadModels = configManager.getLoadModels();
        if (loadModels == null || loadModels.isEmpty()) {
            return;
        }

        ModelManager modelManager = ModelManager.getInstance();
        List<String> urls;
        if ("ALL".equalsIgnoreCase(loadModels)) {
            String modelStore = configManager.getModelStore();
            if (modelStore == null) {
                logger.warn("Model store is not configured.");
                return;
            }

            Path modelStoreDir = Paths.get(modelStore);
            if (!Files.isDirectory(modelStoreDir)) {
                logger.warn("Model store path is not found: {}", modelStore);
                return;
            }

            // Check folders to see if they can be models as well
            urls =
                    Files.list(modelStoreDir)
                            .filter(
                                    p -> {
                                        try {
                                            return !Files.isHidden(p) && Files.isDirectory(p)
                                                    || FilenameUtils.isArchiveFile(p.toString());
                                        } catch (IOException e) {
                                            logger.warn("Failed to access file: " + p, e);
                                            return false;
                                        }
                                    })
                            .map(
                                    p -> {
                                        try {
                                            return p.toUri().toURL().toString();
                                        } catch (MalformedURLException e) {
                                            throw new AssertionError("Invalid path: " + p, e);
                                        }
                                    })
                            .collect(Collectors.toList());
        } else {
            String[] modelsUrls = loadModels.split("[, ]+");
            urls = Arrays.asList(modelsUrls);
        }

        for (String url : urls) {
            int workers = configManager.getDefaultWorkers();
            CompletableFuture<ModelInfo> future = modelManager.registerModel(null, url, 1, 100);
            ModelInfo info = future.join();
            String modelName = info.getModelName();
            modelManager.updateModel(modelName, workers, workers);
            startupModels.add(modelName);
        }
    }

    private static void printHelp(String msg, Options options) {
        HelpFormatter formatter = new HelpFormatter();
        formatter.setLeftPadding(1);
        formatter.setWidth(120);
        formatter.printHelp(msg, options);
    }
}

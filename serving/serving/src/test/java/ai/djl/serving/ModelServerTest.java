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

import ai.djl.modality.Classifications.Classification;
import ai.djl.serving.http.DescribeModelResponse;
import ai.djl.serving.http.ErrorResponse;
import ai.djl.serving.http.ListModelsResponse;
import ai.djl.serving.http.StatusResponse;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.Connector;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;
import ai.djl.util.cuda.CudaUtils;
import com.google.gson.reflect.TypeToken;
import io.netty.bootstrap.Bootstrap;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import io.netty.channel.Channel;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelOption;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.DefaultFullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpClientCodec;
import io.netty.handler.codec.http.HttpContentDecompressor;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpHeaders;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpRequest;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.multipart.HttpPostRequestEncoder;
import io.netty.handler.codec.http.multipart.MemoryFileUpload;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.util.InsecureTrustManagerFactory;
import io.netty.handler.stream.ChunkedWriteHandler;
import io.netty.handler.timeout.ReadTimeoutHandler;
import io.netty.util.internal.logging.InternalLoggerFactory;
import io.netty.util.internal.logging.Slf4JLoggerFactory;
import java.io.IOException;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.lang.reflect.Type;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.GeneralSecurityException;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import javax.net.ssl.HttpsURLConnection;
import javax.net.ssl.SSLContext;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterSuite;
import org.testng.annotations.BeforeSuite;
import org.testng.annotations.Test;

public class ModelServerTest {

    private static final String ERROR_NOT_FOUND =
            "Requested resource is not found, please refer to API document.";
    private static final String ERROR_METHOD_NOT_ALLOWED =
            "Requested method is not allowed, please refer to API document.";

    private ConfigManager configManager;
    private ModelServer server;
    private byte[] testImage;
    volatile CountDownLatch latch;
    volatile HttpResponseStatus httpStatus;
    volatile String result;
    volatile HttpHeaders headers;

    static {
        try {
            SSLContext context = SSLContext.getInstance("TLS");
            context.init(null, InsecureTrustManagerFactory.INSTANCE.getTrustManagers(), null);

            HttpsURLConnection.setDefaultSSLSocketFactory(context.getSocketFactory());

            HttpsURLConnection.setDefaultHostnameVerifier((s, sslSession) -> true);
        } catch (GeneralSecurityException ignore) {
            // ignore
        }
    }

    @BeforeSuite
    public void beforeSuite()
            throws InterruptedException, IOException, GeneralSecurityException, ParseException {
        Path imageFile = Paths.get("../../examples/src/test/resources/0.png");
        try (InputStream is = Files.newInputStream(imageFile)) {
            testImage = Utils.toByteArray(is);
        }

        String[] args = {"-f", "src/test/resources/config.properties"};
        Arguments arguments = ConfigManagerTest.parseArguments(args);
        Assert.assertFalse(arguments.hasHelp());

        ConfigManager.init(arguments);
        configManager = ConfigManager.getInstance();

        InternalLoggerFactory.setDefaultFactory(Slf4JLoggerFactory.INSTANCE);

        server = new ModelServer(configManager);
        server.start();
    }

    @AfterSuite
    public void afterSuite() {
        server.stop();
    }

    @Test
    public void test()
            throws InterruptedException, HttpPostRequestEncoder.ErrorDataEncoderException,
                    IOException, ParseException, GeneralSecurityException,
                    ReflectiveOperationException {
        Assert.assertTrue(server.isRunning());

        Channel channel = null;
        for (int i = 0; i < 5; ++i) {
            try {
                channel = connect(Connector.ConnectorType.MANAGEMENT);
                break;
            } catch (AssertionError e) {
                Thread.sleep(100);
            }
        }

        Assert.assertNotNull(channel, "Failed to connect to inference port.");

        // inference API
        testPing(channel);
        testRoot(channel);
        testPredictions(channel);
        testInvocations(channel);
        testInvocationsMultipart(channel);
        testDescribeApi(channel);

        // management API
        testRegisterModel(channel);
        testRegisterModelAsync(channel);
        testScaleModel(channel);
        testDescribeModel(channel);
        testUnregisterModel(channel);

        testPredictionsInvalidRequestSize(channel);

        // plugin tests
        //   testStaticHtmlRequest();

        channel.close().sync();

        // negative test case that channel will be closed by server
        testInvalidUri();
        testInvalidPredictionsUri();
        testInvalidPredictionsMethod();
        testPredictionsModelNotFound();
        testInvalidDescribeModel();
        testDescribeModelNotFound();
        testInvalidManagementUri();
        testInvalidManagementMethod();
        testUnregisterModelNotFound();
        testInvalidScaleModel();
        testScaleModelNotFound();
        testRegisterModelMissingUrl();
        testRegisterModelNotFound();
        testRegisterModelConflict();
        testServiceUnavailable();

        ConfigManagerTest.testSsl();
    }

    private void testRoot(Channel channel) throws InterruptedException {
        reset();
        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.OPTIONS, "/");
        channel.writeAndFlush(req).sync();
        latch.await();

        Assert.assertEquals(result, "{}\n");
    }

    private void testPing(Channel channel) throws InterruptedException {
        reset();
        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/ping");
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Healthy");
        Assert.assertTrue(headers.contains("x-request-id"));
    }

    private void testPredictions(Channel channel) throws InterruptedException {
        reset();
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/mlp");
        req.content().writeBytes(testImage);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
        channel.writeAndFlush(req);

        latch.await();

        Type type = new TypeToken<List<Classification>>() {}.getType();
        List<Classification> classifications = JsonUtils.GSON.fromJson(result, type);
        Assert.assertEquals(classifications.get(0).getClassName(), "0");
    }

    private void testInvocations(Channel channel) throws InterruptedException {
        reset();
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/invocations");
        req.content().writeBytes(testImage);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
        channel.writeAndFlush(req);
        latch.await();

        Type type = new TypeToken<List<Classification>>() {}.getType();
        List<Classification> classifications = JsonUtils.GSON.fromJson(result, type);
        Assert.assertEquals(classifications.get(0).getClassName(), "0");
    }

    private void testInvocationsMultipart(Channel channel)
            throws InterruptedException, HttpPostRequestEncoder.ErrorDataEncoderException,
                    IOException {
        reset();
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/invocations?model_name=mlp");

        ByteBuf content = Unpooled.buffer(testImage.length);
        content.writeBytes(testImage);
        HttpPostRequestEncoder encoder = new HttpPostRequestEncoder(req, true);
        encoder.addBodyAttribute("test", "test");
        MemoryFileUpload body =
                new MemoryFileUpload("data", "0.png", "image/png", null, null, testImage.length);
        body.setContent(content);
        encoder.addBodyHttpData(body);

        channel.writeAndFlush(encoder.finalizeRequest());
        if (encoder.isChunked()) {
            channel.writeAndFlush(encoder).sync();
        }

        latch.await();

        Type type = new TypeToken<List<Classification>>() {}.getType();
        List<Classification> classifications = JsonUtils.GSON.fromJson(result, type);
        Assert.assertEquals(classifications.get(0).getClassName(), "0");
    }

    private void testRegisterModelAsync(Channel channel)
            throws InterruptedException, UnsupportedEncodingException {
        reset();
        String url = "https://resources.djl.ai/test-models/mlp.tar.gz";
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?model_name=mlp_1&synchronous=false&url="
                                + URLEncoder.encode(url, StandardCharsets.UTF_8.name()));
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse statusResp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(statusResp.getStatus(), "Model \"mlp_1\" registration scheduled.");

        boolean modelRegistered = false;
        OUTER:
        for (int i = 0; i < 5; ++i) {
            String token = "";
            while (token != null) {
                reset();
                req =
                        new DefaultFullHttpRequest(
                                HttpVersion.HTTP_1_1,
                                HttpMethod.GET,
                                "/models?limit=1&next_page_token=" + token);
                channel.writeAndFlush(req);
                latch.await();

                ListModelsResponse resp = JsonUtils.GSON.fromJson(result, ListModelsResponse.class);
                for (ListModelsResponse.ModelItem item : resp.getModels()) {
                    Assert.assertNotNull(item.getModelUrl());
                    if ("mlp_1".equals(item.getModelName())) {
                        modelRegistered = true;
                        break OUTER;
                    }
                }
                token = resp.getNextPageToken();
            }
            Thread.sleep(100);
        }
        Assert.assertTrue(modelRegistered);
    }

    private void testRegisterModel(Channel channel)
            throws InterruptedException, UnsupportedEncodingException {
        reset();

        String url = "https://resources.djl.ai/test-models/mlp.tar.gz";
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?model_name=mlp_2&url="
                                + URLEncoder.encode(url, StandardCharsets.UTF_8.name()));
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Model \"mlp_2\" registered.");
    }

    private void testScaleModel(Channel channel) throws InterruptedException {
        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.PUT,
                        "/models/mlp_2?min_worker=2&max_worker=4");
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(
                resp.getStatus(),
                "Model \"mlp_2\" worker scaled. New Worker configuration min workers:2 max workers:4");
    }

    private void testDescribeModel(Channel channel) throws InterruptedException {
        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/models/mlp_2");
        channel.writeAndFlush(req);
        latch.await();

        DescribeModelResponse resp = JsonUtils.GSON.fromJson(result, DescribeModelResponse.class);
        Assert.assertTrue(resp.getWorkers().size() > 1);

        Assert.assertEquals(resp.getModelName(), "mlp_2");
        Assert.assertNotNull(resp.getModelUrl());
        Assert.assertEquals(resp.getMinWorkers(), 2);
        Assert.assertEquals(resp.getMaxWorkers(), 4);
        Assert.assertEquals(resp.getBatchSize(), 1);
        Assert.assertEquals(resp.getMaxBatchDelay(), 100);
        Assert.assertEquals(resp.getStatus(), "Healthy");
        DescribeModelResponse.Worker worker = resp.getWorkers().get(0);
        Assert.assertTrue(worker.getId() > 0);
        Assert.assertNotNull(worker.getStartTime());
        Assert.assertNotNull(worker.getStatus());
        Assert.assertEquals(worker.isGpu(), CudaUtils.hasCuda());
    }

    private void testUnregisterModel(Channel channel) throws InterruptedException {
        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.DELETE, "/models/mlp_1");
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Model \"mlp_1\" unregistered");
    }

    private void testDescribeApi(Channel channel) throws InterruptedException {
        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.OPTIONS, "/predictions/mlp");
        channel.writeAndFlush(req);
        latch.await();

        Assert.assertEquals(result, "{}\n");
    }

    private void testStaticHtmlRequest() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.INFERENCE);
        Assert.assertNotNull(channel);

        reset();
        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/");
        channel.writeAndFlush(req).sync();
        latch.await();

        Assert.assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());
    }

    private void testPredictionsInvalidRequestSize(Channel channel) throws InterruptedException {
        reset();
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/mlp");

        req.content().writeZero(11485760);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
        channel.writeAndFlush(req);

        latch.await();

        Assert.assertEquals(httpStatus, HttpResponseStatus.REQUEST_ENTITY_TOO_LARGE);
    }

    private void testInvalidUri() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.INFERENCE);
        Assert.assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/InvalidUrl");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
            Assert.assertEquals(resp.getMessage(), ERROR_NOT_FOUND);
        }
    }

    private void testInvalidDescribeModel() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.INFERENCE);
        Assert.assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.OPTIONS, "/predictions/InvalidModel");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
            Assert.assertEquals(resp.getMessage(), "Model not found: InvalidModel");
        }
    }

    private void testInvalidPredictionsUri() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.INFERENCE);
        Assert.assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/predictions");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
            Assert.assertEquals(resp.getMessage(), ERROR_NOT_FOUND);
        }
    }

    private void testPredictionsModelNotFound() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.INFERENCE);
        Assert.assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/predictions/InvalidModel");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
            Assert.assertEquals(resp.getMessage(), "Model not found: InvalidModel");
        }
    }

    private void testInvalidManagementUri() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        Assert.assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/InvalidUrl");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
            Assert.assertEquals(resp.getMessage(), ERROR_NOT_FOUND);
        }
    }

    private void testInvalidManagementMethod() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        Assert.assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, "/models");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            Assert.assertEquals(resp.getCode(), HttpResponseStatus.METHOD_NOT_ALLOWED.code());
            Assert.assertEquals(resp.getMessage(), ERROR_METHOD_NOT_ALLOWED);
        }
    }

    private void testInvalidPredictionsMethod() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        Assert.assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/models/noop");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            Assert.assertEquals(resp.getCode(), HttpResponseStatus.METHOD_NOT_ALLOWED.code());
            Assert.assertEquals(resp.getMessage(), ERROR_METHOD_NOT_ALLOWED);
        }
    }

    private void testDescribeModelNotFound() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        Assert.assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/models/InvalidModel");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
            Assert.assertEquals(resp.getMessage(), "Model not found: InvalidModel");
        }
    }

    private void testRegisterModelMissingUrl() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        Assert.assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/models");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            Assert.assertEquals(resp.getCode(), HttpResponseStatus.BAD_REQUEST.code());
            Assert.assertEquals(resp.getMessage(), "Parameter url is required.");
        }
    }

    private void testRegisterModelNotFound() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        Assert.assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/models?url=InvalidUrl");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
            Assert.assertEquals(
                    resp.getMessage(), "No matching model with specified Input/Output type found.");
        }
    }

    private void testRegisterModelConflict()
            throws InterruptedException, UnsupportedEncodingException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        Assert.assertNotNull(channel);

        reset();
        String url = "https://resources.djl.ai/test-models/mlp.tar.gz";
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?model_name=mlp_2&url="
                                + URLEncoder.encode(url, StandardCharsets.UTF_8.name()));
        channel.writeAndFlush(req);
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            Assert.assertEquals(resp.getCode(), HttpResponseStatus.BAD_REQUEST.code());
            Assert.assertEquals(resp.getMessage(), "Model mlp_2 is already registered.");
        }
    }

    private void testInvalidScaleModel() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        Assert.assertNotNull(channel);

        reset();
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.PUT,
                        "/models/mlp?min_worker=10&max_worker=1");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            Assert.assertEquals(httpStatus, HttpResponseStatus.BAD_REQUEST);
            Assert.assertEquals(resp.getMessage(), "max_worker cannot be less than min_worker.");
        }
    }

    private void testScaleModelNotFound() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        Assert.assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, "/models/fake");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
            Assert.assertEquals(resp.getMessage(), "Model not found: fake");
        }
    }

    private void testUnregisterModelNotFound() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        Assert.assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.DELETE, "/models/fake");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
            Assert.assertEquals(resp.getMessage(), "Model not found: fake");
        }
    }

    private void testServiceUnavailable() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        Assert.assertNotNull(channel);

        reset();
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.PUT,
                        "/models/mlp_2?min_worker=0&max_worker=0");
        channel.writeAndFlush(req);
        latch.await();

        reset();
        req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/mlp_2");
        req.content().writeBytes(testImage);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            Assert.assertEquals(resp.getCode(), HttpResponseStatus.SERVICE_UNAVAILABLE.code());
            Assert.assertEquals(
                    resp.getMessage(), "No worker is available to serve request: mlp_2");
        }
    }

    private Channel connect(Connector.ConnectorType type) {
        Logger logger = LoggerFactory.getLogger(ModelServerTest.class);

        final Connector connector = configManager.getConnector(type);
        try {
            Bootstrap b = new Bootstrap();
            final SslContext sslCtx =
                    SslContextBuilder.forClient()
                            .trustManager(InsecureTrustManagerFactory.INSTANCE)
                            .build();
            b.group(Connector.newEventLoopGroup(1))
                    .channel(connector.getClientChannel())
                    .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 10000)
                    .handler(
                            new ChannelInitializer<Channel>() {

                                /** {@inheritDoc} */
                                @Override
                                public void initChannel(Channel ch) {
                                    ChannelPipeline p = ch.pipeline();
                                    if (connector.isSsl()) {
                                        p.addLast(sslCtx.newHandler(ch.alloc()));
                                    }
                                    p.addLast(new ReadTimeoutHandler(30));
                                    p.addLast(new HttpClientCodec());
                                    p.addLast(new HttpContentDecompressor());
                                    p.addLast(new ChunkedWriteHandler());
                                    p.addLast(new HttpObjectAggregator(6553600));
                                    p.addLast(new TestHandler());
                                }
                            });

            return b.connect(connector.getSocketAddress()).sync().channel();
        } catch (Throwable t) {
            logger.warn("Connect error.", t);
        }
        throw new AssertionError("Failed connect to model server.");
    }

    private void reset() {
        result = null;
        httpStatus = null;
        headers = null;
        latch = new CountDownLatch(1);
    }

    @ChannelHandler.Sharable
    private class TestHandler extends SimpleChannelInboundHandler<FullHttpResponse> {

        /** {@inheritDoc} */
        @Override
        public void channelRead0(ChannelHandlerContext ctx, FullHttpResponse msg) {
            httpStatus = msg.status();
            result = msg.content().toString(StandardCharsets.UTF_8);
            headers = msg.headers();
            latch.countDown();
        }

        /** {@inheritDoc} */
        @Override
        public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
            Logger logger = LoggerFactory.getLogger(TestHandler.class);
            logger.error("Unknown exception", cause);
            ctx.close();
            latch.countDown();
        }
    }
}

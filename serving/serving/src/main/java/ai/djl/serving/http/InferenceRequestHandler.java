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
package ai.djl.serving.http;

import ai.djl.ModelException;
import ai.djl.modality.Input;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.NettyUtils;
import ai.djl.serving.wlm.Job;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.ModelManager;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.nio.charset.StandardCharsets;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A class handling inbound HTTP requests for the management API. */
public class InferenceRequestHandler extends HttpRequestHandler {

    private static final Logger logger = LoggerFactory.getLogger(InferenceRequestHandler.class);

    private RequestParser requestParser;

    private static final Pattern PATTERN =
            Pattern.compile("^/(ping|invocations|predictions)([/?].*)?");

    /** default constructor. */
    public InferenceRequestHandler() {
        this.requestParser = new RequestParser();
    }

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) throws Exception {
        if (super.acceptInboundMessage(msg)) {
            FullHttpRequest req = (FullHttpRequest) msg;
            return PATTERN.matcher(req.uri()).matches();
        }
        return false;
    }

    /** {@inheritDoc} */
    @Override
    protected void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException {
        switch (segments[1]) {
            case "ping":
                // TODO: Check if its OK to send other 2xx errors to ALB for "Partial Healthy"
                // and "Unhealthy"
                ModelManager.getInstance()
                        .workerStatus()
                        .thenAccept(
                                response ->
                                        NettyUtils.sendJsonResponse(
                                                ctx,
                                                new StatusResponse(response),
                                                HttpResponseStatus.OK));
                break;
            case "invocations":
                handleInvocations(ctx, req, decoder);
                break;
            case "predictions":
                handlePredictions(ctx, req, decoder, segments);
                break;
            default:
                throw new AssertionError("Invalid request uri: " + req.uri());
        }
    }

    private void handlePredictions(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelNotFoundException {
        if (segments.length < 3) {
            throw new ResourceNotFoundException();
        }
        Input input = requestParser.parseRequest(ctx, req, decoder);
        predict(ctx, req, input, segments[2]);
    }

    private void handleInvocations(
            ChannelHandlerContext ctx, FullHttpRequest req, QueryStringDecoder decoder)
            throws ModelNotFoundException {
        Input input = requestParser.parseRequest(ctx, req, decoder);
        String modelName = NettyUtils.getParameter(decoder, "model_name", null);
        if ((modelName == null || modelName.isEmpty())) {
            modelName = input.getProperty("model_name", null);
            if (modelName == null) {
                byte[] buf = input.getContent().get("model_name");
                if (buf != null) {
                    modelName = new String(buf, StandardCharsets.UTF_8);
                }
            }
        }
        if (modelName == null) {
            if (ModelManager.getInstance().getStartupModels().size() == 1) {
                modelName = ModelManager.getInstance().getStartupModels().iterator().next();
            }
            if (modelName == null) {
                throw new BadRequestException("Parameter model_name is required.");
            }
        }
        predict(ctx, req, input, modelName);
    }

    private void predict(
            ChannelHandlerContext ctx, FullHttpRequest req, Input input, String modelName)
            throws ModelNotFoundException {
        ModelManager modelManager = ModelManager.getInstance();
        ModelInfo model = modelManager.getModels().get(modelName);
        if (model == null) {
            String regex = ConfigManager.getInstance().getModelUrlPattern();
            if (regex == null) {
                throw new ModelNotFoundException("Model not found: " + modelName);
            }
            String modelUrl = input.getProperty("model_url", null);
            if (modelUrl == null) {
                byte[] buf = input.getContent().get("model_url");
                if (buf == null) {
                    throw new ModelNotFoundException("Parameter model_url is required.");
                }
                modelUrl = new String(buf, StandardCharsets.UTF_8);
                if (!modelUrl.matches(regex)) {
                    throw new ModelNotFoundException("Permission denied: " + modelUrl);
                }
            }
            String engineName = input.getProperty("engine_name", null);

            logger.info("Loading model {} from: {}", modelName, modelUrl);

            modelManager
                    .registerModel(
                            modelName,
                            modelUrl,
                            engineName,
                            ConfigManager.getInstance().getBatchSize(),
                            ConfigManager.getInstance().getMaxBatchDelay(),
                            ConfigManager.getInstance().getMaxIdleTime())
                    .thenAccept(m -> modelManager.triggerModelUpdated(m.scaleWorkers(1, 1)))
                    .thenAccept(
                            p -> {
                                try {
                                    if (!modelManager.addJob(new Job(ctx, modelName, input))) {
                                        throw new ServiceUnavailableException(
                                                "No worker is available to serve request: "
                                                        + modelName);
                                    }
                                } catch (ModelNotFoundException e) {
                                    logger.warn("Unexpected error", e);
                                    NettyUtils.sendError(ctx, e);
                                }
                            })
                    .exceptionally(
                            t -> {
                                logger.warn("Unexpected error", t);
                                NettyUtils.sendError(ctx, t);
                                return null;
                            });
            return;
        }

        if (HttpMethod.OPTIONS.equals(req.method())) {
            NettyUtils.sendJsonResponse(ctx, "{}");
            return;
        }

        Job job = new Job(ctx, modelName, input);
        if (!ModelManager.getInstance().addJob(job)) {
            logger.error("unable to process prediction. no free worker available.");
            throw new ServiceUnavailableException(
                    "No worker is available to serve request: " + modelName);
        }
    }
}

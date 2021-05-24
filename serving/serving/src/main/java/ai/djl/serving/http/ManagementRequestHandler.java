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
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.serving.util.NettyUtils;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.ModelManager;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.regex.Pattern;

/**
 * A class handling inbound HTTP requests to the management API.
 *
 * <p>This class
 */
public class ManagementRequestHandler extends HttpRequestHandler {

    /** HTTP Parameter "synchronous". */
    private static final String SYNCHRONOUS_PARAMETER = "synchronous";
    /** HTTP Parameter "initial_workers". */
    private static final String INITIAL_WORKERS_PARAMETER = "initial_workers";
    /** HTTP Parameter "url". */
    private static final String URL_PARAMETER = "url";
    /** HTTP Parameter "batch_size". */
    private static final String BATCH_SIZE_PARAMETER = "batch_size";
    /** HTTP Parameter "model_name". */
    private static final String MODEL_NAME_PARAMETER = "model_name";
    /** HTTP Parameter "model_name". */
    private static final String ENGINE_NAME_PARAMETER = "engine_name";
    /** HTTP Parameter "max_batch_delay". */
    private static final String MAX_BATCH_DELAY_PARAMETER = "max_batch_delay";
    /** HTTP Parameter "max_idle_time". */
    private static final String MAX_IDLE_TIME__PARAMETER = "max_idle_time";
    /** HTTP Parameter "max_worker". */
    private static final String MAX_WORKER_PARAMETER = "max_worker";
    /** HTTP Parameter "min_worker". */
    private static final String MIN_WORKER_PARAMETER = "min_worker";

    private static final Pattern PATTERN = Pattern.compile("^/models([/?].*)?");

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
        HttpMethod method = req.method();
        if (segments.length < 3) {
            if (HttpMethod.GET.equals(method)) {
                handleListModels(ctx, decoder);
                return;
            } else if (HttpMethod.POST.equals(method)) {
                handleRegisterModel(ctx, decoder);
                return;
            }
            throw new MethodNotAllowedException();
        }

        if (HttpMethod.GET.equals(method)) {
            handleDescribeModel(ctx, segments[2]);
        } else if (HttpMethod.PUT.equals(method)) {
            handleScaleModel(ctx, decoder, segments[2]);
        } else if (HttpMethod.DELETE.equals(method)) {
            handleUnregisterModel(ctx, segments[2]);
        } else {
            throw new MethodNotAllowedException();
        }
    }

    private void handleListModels(ChannelHandlerContext ctx, QueryStringDecoder decoder) {
        int limit = NettyUtils.getIntParameter(decoder, "limit", 100);
        int pageToken = NettyUtils.getIntParameter(decoder, "next_page_token", 0);
        if (limit > 100 || limit < 0) {
            limit = 100;
        }
        if (pageToken < 0) {
            pageToken = 0;
        }

        ModelManager modelManager = ModelManager.getInstance();
        Map<String, ModelInfo> models = modelManager.getModels();

        List<String> keys = new ArrayList<>(models.keySet());
        Collections.sort(keys);
        ListModelsResponse list = new ListModelsResponse();

        int last = pageToken + limit;
        if (last > keys.size()) {
            last = keys.size();
        } else {
            list.setNextPageToken(String.valueOf(last));
        }

        for (int i = pageToken; i < last; ++i) {
            String modelName = keys.get(i);
            ModelInfo model = models.get(modelName);
            list.addModel(modelName, model.getModelUrl());
        }

        NettyUtils.sendJsonResponse(ctx, list);
    }

    private void handleDescribeModel(ChannelHandlerContext ctx, String modelName)
            throws ModelNotFoundException {
        ModelManager modelManager = ModelManager.getInstance();
        DescribeModelResponse resp = modelManager.describeModel(modelName);
        NettyUtils.sendJsonResponse(ctx, resp);
    }

    private void handleRegisterModel(final ChannelHandlerContext ctx, QueryStringDecoder decoder) {
        String modelUrl = NettyUtils.getParameter(decoder, URL_PARAMETER, null);
        if (modelUrl == null) {
            throw new BadRequestException("Parameter url is required.");
        }

        String modelName = NettyUtils.getParameter(decoder, MODEL_NAME_PARAMETER, null);
        if (modelName == null || modelName.isEmpty()) {
            modelName = ModelInfo.inferModelNameFromUrl(modelUrl);
        }
        String engineName = NettyUtils.getParameter(decoder, ENGINE_NAME_PARAMETER, null);
        int batchSize = NettyUtils.getIntParameter(decoder, BATCH_SIZE_PARAMETER, 1);
        int maxBatchDelay = NettyUtils.getIntParameter(decoder, MAX_BATCH_DELAY_PARAMETER, 100);
        int maxIdleTime = NettyUtils.getIntParameter(decoder, MAX_IDLE_TIME__PARAMETER, 60);
        final int initialWorkers =
                NettyUtils.getIntParameter(decoder, INITIAL_WORKERS_PARAMETER, 1);
        boolean synchronous =
                Boolean.parseBoolean(
                        NettyUtils.getParameter(decoder, SYNCHRONOUS_PARAMETER, "true"));

        final ModelManager modelManager = ModelManager.getInstance();
        CompletableFuture<ModelInfo> future =
                modelManager.registerModel(
                        modelName, modelUrl, engineName, batchSize, maxBatchDelay, maxIdleTime);
        CompletableFuture<Void> f =
                future.thenAccept(
                        modelInfo ->
                                modelManager.triggerModelUpdated(
                                        modelInfo
                                                .scaleWorkers(initialWorkers, initialWorkers)
                                                .configurePool(maxIdleTime, maxBatchDelay)
                                                .configureModelBatch(batchSize)));

        if (synchronous) {
            final String msg = "Model \"" + modelName + "\" registered.";
            f = f.thenAccept(m -> NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg)));
        } else {
            String msg = "Model \"" + modelName + "\" registration scheduled.";
            NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
        }

        f.exceptionally(
                t -> {
                    NettyUtils.sendError(ctx, t.getCause());
                    return null;
                });
    }

    private void handleUnregisterModel(ChannelHandlerContext ctx, String modelName)
            throws ModelNotFoundException {
        ModelManager modelManager = ModelManager.getInstance();
        if (!modelManager.unregisterModel(modelName)) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        }
        String msg = "Model \"" + modelName + "\" unregistered";
        NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
    }

    private void handleScaleModel(
            ChannelHandlerContext ctx, QueryStringDecoder decoder, String modelName)
            throws ModelNotFoundException {
        try {

            ModelManager modelManager = ModelManager.getInstance();
            ModelInfo modelInfo = modelManager.getModels().get(modelName);
            if (modelInfo == null) {
                throw new ModelNotFoundException("Model not found: " + modelName);
            }
            int minWorkers =
                    NettyUtils.getIntParameter(
                            decoder, MIN_WORKER_PARAMETER, modelInfo.getMinWorkers());
            int maxWorkers =
                    NettyUtils.getIntParameter(
                            decoder, MAX_WORKER_PARAMETER, modelInfo.getMaxWorkers());
            if (maxWorkers < minWorkers) {
                throw new BadRequestException("max_worker cannot be less than min_worker.");
            }

            int maxIdleTime =
                    NettyUtils.getIntParameter(
                            decoder, MAX_IDLE_TIME__PARAMETER, modelInfo.getMaxIdleTime());
            int maxBatchDelay =
                    NettyUtils.getIntParameter(
                            decoder, MAX_BATCH_DELAY_PARAMETER, modelInfo.getMaxBatchDelay());

            modelInfo =
                    modelInfo
                            .scaleWorkers(minWorkers, maxWorkers)
                            .configurePool(maxIdleTime, maxBatchDelay);
            modelManager.triggerModelUpdated(modelInfo);

            String msg =
                    "Model \""
                            + modelName
                            + "\" worker scaled. New Worker configuration min workers:"
                            + minWorkers
                            + " max workers:"
                            + maxWorkers;
            NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
        } catch (NumberFormatException ex) {
            throw new BadRequestException("parameter is invalid number." + ex.getMessage(), ex);
        }
    }
}

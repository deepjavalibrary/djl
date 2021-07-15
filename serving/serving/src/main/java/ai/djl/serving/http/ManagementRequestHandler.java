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
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.NettyUtils;
import ai.djl.serving.wlm.Endpoint;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.ModelManager;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponseStatus;
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
    /** HTTP Parameter "url". */
    private static final String URL_PARAMETER = "url";
    /** HTTP Parameter "batch_size". */
    private static final String BATCH_SIZE_PARAMETER = "batch_size";
    /** HTTP Parameter "model_name". */
    private static final String MODEL_NAME_PARAMETER = "model_name";
    /** HTTP Parameter "model_version". */
    private static final String MODEL_VERSION_PARAMETER = "model_version";
    /** HTTP Parameter "engine". */
    private static final String ENGINE_NAME_PARAMETER = "engine";
    /** HTTP Parameter "gpu_id". */
    private static final String GPU_ID_PARAMETER = "gpu_id";
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

        String modelName = segments[2];
        String version = null;
        if (segments.length > 3) {
            version = segments[3];
        }

        if (HttpMethod.GET.equals(method)) {
            handleDescribeModel(ctx, modelName, version);
        } else if (HttpMethod.PUT.equals(method)) {
            handleScaleModel(ctx, decoder, modelName, version);
        } else if (HttpMethod.DELETE.equals(method)) {
            handleUnregisterModel(ctx, modelName, version);
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
        Map<String, Endpoint> endpoints = modelManager.getEndpoints();

        List<String> keys = new ArrayList<>(endpoints.keySet());
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
            for (ModelInfo m : endpoints.get(modelName).getModels()) {
                list.addModel(modelName, m.getModelUrl());
            }
        }

        NettyUtils.sendJsonResponse(ctx, list);
    }

    private void handleDescribeModel(ChannelHandlerContext ctx, String modelName, String version)
            throws ModelNotFoundException {
        ModelManager modelManager = ModelManager.getInstance();
        DescribeModelResponse resp = modelManager.describeModel(modelName, version);
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
        String version = NettyUtils.getParameter(decoder, MODEL_VERSION_PARAMETER, null);
        int gpuId = NettyUtils.getIntParameter(decoder, GPU_ID_PARAMETER, -1);
        String engineName = NettyUtils.getParameter(decoder, ENGINE_NAME_PARAMETER, null);
        int batchSize = NettyUtils.getIntParameter(decoder, BATCH_SIZE_PARAMETER, 1);
        int maxBatchDelay = NettyUtils.getIntParameter(decoder, MAX_BATCH_DELAY_PARAMETER, 100);
        int maxIdleTime = NettyUtils.getIntParameter(decoder, MAX_IDLE_TIME__PARAMETER, 60);
        int minWorkers = NettyUtils.getIntParameter(decoder, MIN_WORKER_PARAMETER, 1);
        int defaultWorkers = ConfigManager.getInstance().getDefaultWorkers();
        int maxWorkers = NettyUtils.getIntParameter(decoder, MAX_WORKER_PARAMETER, defaultWorkers);
        boolean synchronous =
                Boolean.parseBoolean(
                        NettyUtils.getParameter(decoder, SYNCHRONOUS_PARAMETER, "true"));

        final ModelManager modelManager = ModelManager.getInstance();
        CompletableFuture<ModelInfo> future =
                modelManager.registerModel(
                        modelName,
                        version,
                        modelUrl,
                        engineName,
                        gpuId,
                        batchSize,
                        maxBatchDelay,
                        maxIdleTime);
        CompletableFuture<Void> f =
                future.thenAccept(
                        m ->
                                modelManager.triggerModelUpdated(
                                        m.scaleWorkers(minWorkers, maxWorkers)
                                                .configurePool(maxIdleTime)
                                                .configureModelBatch(batchSize, maxBatchDelay)));

        if (synchronous) {
            final String msg = "Model \"" + modelName + "\" registered.";
            f = f.thenAccept(m -> NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg)));
        } else {
            String msg = "Model \"" + modelName + "\" registration scheduled.";
            NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg), HttpResponseStatus.ACCEPTED);
        }

        f.exceptionally(
                t -> {
                    NettyUtils.sendError(ctx, t.getCause());
                    return null;
                });
    }

    private void handleUnregisterModel(ChannelHandlerContext ctx, String modelName, String version)
            throws ModelNotFoundException {
        ModelManager modelManager = ModelManager.getInstance();
        if (!modelManager.unregisterModel(modelName, version)) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        }
        String msg = "Model \"" + modelName + "\" unregistered";
        NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
    }

    private void handleScaleModel(
            ChannelHandlerContext ctx, QueryStringDecoder decoder, String modelName, String version)
            throws ModelNotFoundException {
        try {
            ModelManager modelManager = ModelManager.getInstance();
            ModelInfo modelInfo = modelManager.getModel(modelName, version, false);
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
            int batchSize =
                    NettyUtils.getIntParameter(
                            decoder, BATCH_SIZE_PARAMETER, modelInfo.getBatchSize());
            int maxBatchDelay =
                    NettyUtils.getIntParameter(
                            decoder, MAX_BATCH_DELAY_PARAMETER, modelInfo.getMaxBatchDelay());

            modelInfo
                    .scaleWorkers(minWorkers, maxWorkers)
                    .configurePool(maxIdleTime)
                    .configureModelBatch(batchSize, maxBatchDelay);
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

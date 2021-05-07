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
package ai.djl.serving.central.handler;

import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.serving.central.model.dto.ModelReferenceDTO;
import ai.djl.serving.plugins.RequestHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A handler to handler model meta data requests.
 *
 * @author erik.bamberg@web.de
 */
public class ModelListMetaDataHandler
        implements RequestHandler<CompletableFuture<Map<String, List<ModelReferenceDTO>>>> {

    private static final Logger logger = LoggerFactory.getLogger(ModelListMetaDataHandler.class);

    private static final Pattern URL_PATTERN = Pattern.compile("^/modelzoo/models$");

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) {
        FullHttpRequest request = (FullHttpRequest) msg;

        String uri = request.uri();
        return URL_PATTERN.matcher(uri).matches();
    }

    /** {@inheritDoc} */
    @Override
    public CompletableFuture<Map<String, List<ModelReferenceDTO>>> handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments) {
        logger.info("request models.");
        return CompletableFuture.supplyAsync(
                () -> {
                    try {
                        return ModelZoo.listModels()
                                .entrySet()
                                .stream()
                                .collect(
                                        Collectors.toMap(
                                                e -> e.getKey().getPath(),
                                                e ->
                                                        e.getValue()
                                                                .stream()
                                                                .map(a -> new ModelReferenceDTO(a))
                                                                .collect(Collectors.toList())));
                    } catch (IOException | ModelNotFoundException ex) {
                        throw new IllegalArgumentException(ex.getMessage(), ex);
                    }
                });
    }
}

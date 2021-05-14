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

import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.serving.central.model.dto.DataTransferObjectFactory;
import ai.djl.serving.central.model.dto.ModelDTO;
import ai.djl.serving.plugins.RequestHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.io.IOException;
import java.util.concurrent.CompletableFuture;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A handler to handler model meta data requests.
 *
 * @author erik.bamberg@web.de
 */
public class ModelMetaDataHandler implements RequestHandler<CompletableFuture<ModelDTO>> {

    private static final Logger logger = LoggerFactory.getLogger(ModelMetaDataHandler.class);

    private static final Pattern URL_PATTERN =
            Pattern.compile("/modelzoo/models/([a-zA-Z0-9.:@_-]+)/?");

    private DataTransferObjectFactory dtoFactory;

    /** constructs the ModelMetaDataHandler. */
    public ModelMetaDataHandler() {
        dtoFactory = new DataTransferObjectFactory();
    }

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) {
        FullHttpRequest request = (FullHttpRequest) msg;

        String uri = request.uri();
        return URL_PATTERN.matcher(uri).matches();
    }

    /** {@inheritDoc} */
    @Override
    public CompletableFuture<ModelDTO> handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments) {

        Matcher matcher = URL_PATTERN.matcher(req.uri());
        if (matcher.matches()) {
            String modelParameter = matcher.group(1);
            String[] modelIdent = modelParameter.split(":", -1);
            logger.debug("loading model details for {}", modelParameter);
            String groupId = modelIdent[0];
            String artifactId = modelIdent[1];
            String version = modelIdent[2];
            String modelName = modelIdent[3];

            return CompletableFuture.supplyAsync(
                    () -> {
                        try {
                            Criteria<?, ?> criteria =
                                    Criteria.builder()
                                            .optGroupId(groupId)
                                            .optArtifactId(artifactId)
                                            .optModelName(modelName)
                                            .build();
                            return ModelZoo.listModels(criteria)
                                    .values()
                                    .stream()
                                    .flatMap(each -> each.stream())
                                    .filter(
                                            a ->
                                                    modelName.equals(a.getName())
                                                            && version.equals(a.getVersion()))
                                    .map(artifact -> dtoFactory.create(artifact))
                                    .findFirst()
                                    .orElseThrow(
                                            () ->
                                                    new ModelNotFoundException(
                                                            "model " + modelName + " not found."));

                        } catch (IOException | ModelNotFoundException ex) {
                            throw new IllegalArgumentException(ex.getMessage(), ex);
                        }
                    });
        } else {
            throw new IllegalArgumentException("No Model found in uri");
        }
    }
}

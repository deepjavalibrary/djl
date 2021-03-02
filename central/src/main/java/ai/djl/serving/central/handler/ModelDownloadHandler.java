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

import ai.djl.Application;
import ai.djl.repository.Artifact;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.serving.central.responseencoder.HttpRequestResponse;
import ai.djl.serving.central.utils.NettyUtils;
import ai.djl.serving.http.BadRequestException;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;
import java.util.*;
import java.util.concurrent.CompletableFuture;

final class ModelLink {

    private static final Logger logger = LoggerFactory.getLogger(ModelLink.class);
    private static Map<String, URI> links = new HashMap<String,URI>();
    private ModelLink() {}

    private static URI BASE_URI = URI.create("https://mlrepo.djl.ai/");

    public static Map<String, URI> linkFinder(String modelName) throws IOException, ModelNotFoundException {
        Map<Application, List<Artifact>> models = ModelZoo.listModels();
        models.forEach(
                (app, list) -> {
                    list.forEach(
                            artifact -> {
                                if (artifact.getName().equals(modelName)){
                                    for (Map.Entry<String, Artifact.Item> entry :
                                            artifact.getFiles().entrySet()) {
                                        URI fileUri = URI.create(entry.getValue().getUri());
                                        URI baseUri = artifact.getMetadata().getRepositoryUri();
                                        if (!fileUri.isAbsolute()) {
                                            fileUri = BASE_URI.resolve(baseUri).resolve(fileUri);
                                        }
                                        try {
                                            links.put(entry.getKey(),fileUri);
                                        } catch(Exception e){
                                            logger.info(String.valueOf(e));
                                        }
                                    }
                                }});
                });
        return links;
    }

    public static void main(String[] args) throws IOException, ModelNotFoundException {
        logger.info("Output:");
        logger.info(String.valueOf(linkFinder("simple_pose_resnet50_v1b")));
    }
}


/**
 * A handler to handle deployment requests from the UI/
 * @author erik.bamberg@web.de
 *
 */
public class ModelDownloadHandler extends SimpleChannelInboundHandler<FullHttpRequest> {

    HttpRequestResponse jsonResponse;
    public ModelDownloadHandler() { jsonResponse = new HttpRequestResponse(); }

    /**
     * handle the deployment request by forwarding the request to the serving-instance.
     *
     * @param ctx the context
     * @param request the full request
     */
    @Override
    protected void channelRead0(ChannelHandlerContext ctx, FullHttpRequest request) throws IOException, ModelNotFoundException {
        final Logger logger = LoggerFactory.getLogger(ModelDownloadHandler.class);
        QueryStringDecoder decoder = new QueryStringDecoder(request.uri());
        String modelName=NettyUtils.getParameter(decoder, "modelName", null);
        CompletableFuture.supplyAsync(
                () -> {
                    try {
                        if (modelName!=null) {
                            logger.info(String.valueOf(ModelLink.linkFinder(modelName)));
                            return ModelLink.linkFinder(modelName);
                        } else {
                            throw new BadRequestException("modelName and url is mandatory.");
                        }

                    } catch (IOException | ModelNotFoundException ex) {
                        throw new IllegalArgumentException(ex.getMessage(), ex);
                    }
                })
                .exceptionally((ex) -> Collections.emptyMap())
                .thenAccept(linksMap -> jsonResponse.sendAsJson(ctx, request, linksMap));
    }


    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) {
        FullHttpRequest request = (FullHttpRequest) msg;

        String uri = request.uri();
        return uri.startsWith("/serving/models?");
    }

}

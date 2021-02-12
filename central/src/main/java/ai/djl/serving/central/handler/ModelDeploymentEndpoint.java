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

import ai.djl.serving.central.client.RestCall;
import ai.djl.serving.central.utils.NettyUtils;
import ai.djl.serving.http.BadRequestException;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.util.function.BiConsumer;
import java.util.regex.Pattern;

/**
 * @author erik.bamberg@web.de
 *
 */
public class ModelDeploymentEndpoint  {

    /**
     * curl -X PUT "http://localhost:8080/models?modelName=mlp?min_worker=4&max_worker=12&max_idle_time=60&max_batch_delay=100"
     */
    static final Pattern registerModelPattern=Pattern.compile("^/serving[/?]models$"); // POST
    static final BiConsumer<ChannelHandlerContext,FullHttpRequest> registerModel=(ctx, req)->{ 
        QueryStringDecoder decoder = new QueryStringDecoder(req.uri());
        String modelName=NettyUtils.getParameter(decoder, "modelName", null);
        if (modelName!=null) {
            new RestCall().put("http://localhost:5000/models/modelName=",ctx);
        } else {
            throw new BadRequestException("no modelName found.");
        }
    };

}

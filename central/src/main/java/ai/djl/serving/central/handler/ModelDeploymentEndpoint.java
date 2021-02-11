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
import ai.djl.serving.central.responseencoder.JsonResponse;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.util.CharsetUtil;
import java.util.function.BiConsumer;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.regex.Pattern;

/**
 * @author erik.bamberg@web.de
 *
 */
public class ModelDeploymentEndpoint  {

  
    
    static final Pattern pattern=Pattern.compile("^/serving[/?]models/(.*)?");
    
    static final BiConsumer<ChannelHandlerContext,FullHttpRequest> function=(ctx, req)->{ 
	JsonResponse jsonResponse = new JsonResponse();
	RestCall webClient=new RestCall( (response) -> {
	    jsonResponse.forward(ctx, req, response.content().copy() );
	} );
	webClient.send("http://localhost:8080/models/mlp");
    };

}

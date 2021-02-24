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
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.net.URI;
import java.util.regex.Pattern;

/**
 * A handler to handle deployment requests from the UI/
 * @author erik.bamberg@web.de
 *
 */
public class ModelDeploymentHandler extends SimpleChannelInboundHandler<FullHttpRequest> {

    private static final Pattern PATTERN=Pattern.compile("^/serving[/?]models$");
    private static final HttpMethod METHOD=HttpMethod.POST;
    private String servingInstanceURL;
    /**
     * constructing a ModelDeploymentHandler
     */
    public ModelDeploymentHandler(String servingInstanceURL) {
	this.servingInstanceURL=servingInstanceURL;
    }

    
    /**
     * handle the deployment request by forwarding the request to the serving-instance.
     *
     * @param ctx the context
     * @param request the full request
     */
    @Override
    protected void channelRead0(ChannelHandlerContext ctx, FullHttpRequest request) {	
        QueryStringDecoder decoder = new QueryStringDecoder(request.uri());
        String modelName=NettyUtils.getParameter(decoder, "modelName", null);
        String url=NettyUtils.getParameter(decoder, "url", null);
        if (modelName!=null && url!=null) {
            new RestCall().post(servingInstanceURL+"models?modelName="+modelName+"&url="+RestCall.encodeValue(url),ctx);
        } else {
            throw new BadRequestException("modelName and url is mandatory.");
        }
    }
    
    
    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) throws Exception {
        if (super.acceptInboundMessage(msg)) {
            FullHttpRequest req = (FullHttpRequest) msg;
            URI uri=new URI(req.uri());
            return METHOD.equals(req.method()) && PATTERN.matcher(uri.getPath()).matches();
        }
        return false;
    }
    
}

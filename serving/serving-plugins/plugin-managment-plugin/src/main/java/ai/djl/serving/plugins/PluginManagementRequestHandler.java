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
package ai.djl.serving.plugins;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A HTTP endpoint handler to return information about the loaded plugins.
 *
 * <p>This class is registered by the model-server as a HTTP endpoint handler.
 *
 * @author erik.bamberg@web.de
 */
public class PluginManagementRequestHandler implements RequestHandler<List<PluginMetaData>> {

    private static final Logger logger =
            LoggerFactory.getLogger(PluginManagementRequestHandler.class);

    private PluginManager pluginManager;

    private static final Pattern PATTERN = Pattern.compile("^/(plugins)([/?].*)?");

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) {
        if (msg instanceof FullHttpRequest) {
            FullHttpRequest req = (FullHttpRequest) msg;
            return PATTERN.matcher(req.uri()).matches();
        }
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public List<PluginMetaData> handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments) {
        logger.info("handle plugin management request");
        return new ArrayList<>(pluginManager.listPlugins());
    }

    /**
     * inject the pluginManager from the serving instance.
     *
     * @param pluginManager of the serving instance.
     */
    public void setPluginManager(PluginManager pluginManager) {
        this.pluginManager = pluginManager;
    }
}

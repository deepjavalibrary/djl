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
package ai.djl.serving.http;

import ai.djl.modality.Input;
import ai.djl.serving.util.NettyUtils;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.handler.codec.http.multipart.DefaultHttpDataFactory;
import io.netty.handler.codec.http.multipart.HttpDataFactory;
import io.netty.handler.codec.http.multipart.HttpPostRequestDecoder;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * a parser for inbound request.
 *
 * @author erik.bamberg@web.de
 */
public class RequestParser {

    private static final Logger logger = LoggerFactory.getLogger(RequestParser.class);

    /**
     * parsing a request.
     *
     * @param ctx the context.
     * @param req the full request.
     * @param decoder a decoder to decode the query string.
     * @return parsed input object.
     */
    public Input parseRequest(
            ChannelHandlerContext ctx, FullHttpRequest req, QueryStringDecoder decoder) {
        String requestId = NettyUtils.getRequestId(ctx.channel());
        Input input = new Input(requestId);
        if (decoder != null) {
            for (Map.Entry<String, List<String>> entry : decoder.parameters().entrySet()) {
                String key = entry.getKey();
                for (String value : entry.getValue()) {
                    input.addData(key, value.getBytes(StandardCharsets.UTF_8));
                }
            }
        }

        CharSequence contentType = HttpUtil.getMimeType(req);
        for (Map.Entry<String, String> entry : req.headers().entries()) {
            input.addProperty(entry.getKey(), entry.getValue());
        }

        if (HttpPostRequestDecoder.isMultipart(req)
                || HttpHeaderValues.APPLICATION_X_WWW_FORM_URLENCODED.contentEqualsIgnoreCase(
                        contentType)) {
            HttpDataFactory factory = new DefaultHttpDataFactory(6553500);
            HttpPostRequestDecoder form = new HttpPostRequestDecoder(factory, req);
            try {
                while (form.hasNext()) {
                    NettyUtils.addFormData(form.next(), input);
                }
            } catch (HttpPostRequestDecoder.EndOfDataDecoderException ignore) {
                logger.trace("End of multipart items.");
            } finally {
                form.cleanFiles();
                form.destroy();
            }
        } else {
            byte[] content = NettyUtils.getBytes(req.content());
            input.addData("body", content);
        }
        return input;
    }
}

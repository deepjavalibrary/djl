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
package ai.djl.serving.central.utils;

import ai.djl.modality.Input;
import io.netty.buffer.ByteBuf;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.handler.codec.http.multipart.Attribute;
import io.netty.handler.codec.http.multipart.FileUpload;
import io.netty.handler.codec.http.multipart.InterfaceHttpData;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

/** A utility class that handling Netty request and response. */
public final class NettyUtils {

    private NettyUtils() {}

    /**
     * Returns the bytes for the specified {@code ByteBuf}.
     *
     * @param buf the {@code ByteBuf} to read
     * @return the bytes for the specified {@code ByteBuf}
     */
    public static byte[] getBytes(ByteBuf buf) {
        if (buf.hasArray()) {
            return buf.array();
        }

        byte[] ret = new byte[buf.readableBytes()];
        int readerIndex = buf.readerIndex();
        buf.getBytes(readerIndex, ret);
        return ret;
    }

    /**
     * Reads the parameter's value for the key from the uri.
     *
     * @param decoder the {@code QueryStringDecoder} parsed from uri
     * @param key the parameter key
     * @param def the default value
     * @return the parameter's value
     */
    public static String getParameter(QueryStringDecoder decoder, String key, String def) {
        List<String> param = decoder.parameters().get(key);
        if (param != null && !param.isEmpty()) {
            return param.get(0);
        }
        return def;
    }

    /**
     * Read the parameter's integer value for the key from the uri.
     *
     * @param decoder the {@code QueryStringDecoder} parsed from uri
     * @param key the parameter key
     * @param def the default value
     * @return the parameter's integer value
     * @throws NumberFormatException exception is thrown when the parameter-value is not numeric.
     */
    public static int getIntParameter(QueryStringDecoder decoder, String key, int def) {
        String value = getParameter(decoder, key, null);
        if (value == null || value.isEmpty()) {
            return def;
        }
        return Integer.parseInt(value);
    }

    /**
     * Parses form data and added to the {@link Input} object.
     *
     * @param data the form data
     * @param input the {@link Input} object to be added to
     */
    public static void addFormData(InterfaceHttpData data, Input input) {
        if (data == null) {
            return;
        }
        try {
            String name = data.getName();
            switch (data.getHttpDataType()) {
                case Attribute:
                    Attribute attribute = (Attribute) data;
                    input.addData(name, attribute.getValue().getBytes(StandardCharsets.UTF_8));
                    break;
                case FileUpload:
                    FileUpload fileUpload = (FileUpload) data;
                    input.addData(name, getBytes(fileUpload.getByteBuf()));
                    break;
                default:
                    throw new IllegalArgumentException(
                            "Except form field, but got " + data.getHttpDataType());
            }
        } catch (IOException e) {
            throw new AssertionError(e);
        }
    }
}

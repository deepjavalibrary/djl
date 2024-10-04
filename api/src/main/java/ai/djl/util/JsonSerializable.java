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
package ai.djl.util;

import ai.djl.ndarray.BytesSupplier;

import com.google.gson.JsonElement;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;

import java.io.Serializable;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

/**
 * A class implements {@code JsonSerializable} indicates it can be serialized into a json string.
 */
public interface JsonSerializable extends Serializable, BytesSupplier {

    /**
     * Returns a json presentation of the object.
     *
     * @return a json string
     */
    default String toJson() {
        return JsonUtils.GSON_COMPACT.toJson(serialize());
    }

    /** {@inheritDoc} */
    @Override
    default String getAsString() {
        return toJson();
    }

    /** {@inheritDoc} */
    @Override
    default ByteBuffer toByteBuffer() {
        return ByteBuffer.wrap(toJson().getBytes(StandardCharsets.UTF_8));
    }

    /** {@inheritDoc} */
    JsonElement serialize();

    /** A customized Gson serializer to serialize the {@code Segmentation} object. */
    final class Serializer implements JsonSerializer<JsonSerializable> {

        /** {@inheritDoc} */
        @Override
        public JsonElement serialize(
                JsonSerializable src, Type type, JsonSerializationContext ctx) {
            return src.serialize();
        }
    }
}

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
package ai.djl.repository.responseencoder;

import ai.djl.Application;
import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;
import java.lang.reflect.Type;

/**
 * custom gson serializer to serialize application object.
 *
 * @author erik.bamberg@web.de
 */
public class GSONApplicationSerializer
        implements JsonSerializer<Application>, JsonDeserializer<Application> {

    /**
     * serialize.
     *
     * @see JsonSerializer
     */
    @Override
    public JsonElement serialize(
            Application src, Type typeOfSrc, JsonSerializationContext context) {
        return new JsonPrimitive(src.getPath());
    }

    /**
     * deserialize.
     *
     * @see JsonDeserializer
     */
    @Override
    public Application deserialize(
            JsonElement json, Type typeOfT, JsonDeserializationContext context) {
        return Application.of(json.getAsString());
    }
}

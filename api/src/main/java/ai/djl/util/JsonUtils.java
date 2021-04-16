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
package ai.djl.util;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializer;

/** An interface containing Gson constants. */
public interface JsonUtils {

    Gson GSON = new GsonBuilder().create();
    Gson GSON_PRETTY = builder().create();

    /**
     * Returns a custom {@code GsonBuilder} instance.
     *
     * @return a custom {@code GsonBuilder} instance.
     */
    static GsonBuilder builder() {
        return new GsonBuilder()
                .setDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
                .setPrettyPrinting()
                .registerTypeAdapter(
                        Double.class,
                        (JsonSerializer<Double>)
                                (src, t, ctx) -> {
                                    long v = src.longValue();
                                    if (src.equals(Double.valueOf(String.valueOf(v)))) {
                                        return new JsonPrimitive(v);
                                    }
                                    return new JsonPrimitive(src);
                                });
    }
}

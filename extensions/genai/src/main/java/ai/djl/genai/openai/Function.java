/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.genai.openai;

import ai.djl.genai.FunctionUtils;
import ai.djl.util.JsonUtils;
import ai.djl.util.Pair;
import ai.djl.util.PairList;

import com.google.gson.JsonObject;

import java.lang.reflect.Method;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** A data class represents chat completion schema. */
@SuppressWarnings("MissingJavadocMethod")
public class Function {

    private String name;
    private String description;
    private Object parameters;

    public Function(Builder builder) {
        name = builder.name;
        description = builder.description;
        parameters = builder.parameters;
    }

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    public Object getParameters() {
        return parameters;
    }

    /**
     * Creates a builder to build a {@code Function}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    public static Builder function(Method method) {
        Map<String, Object> parameters = new ConcurrentHashMap<>();
        PairList<String, String> pairs = FunctionUtils.getParameters(method);
        Map<String, Map<String, String>> properties = new ConcurrentHashMap<>();
        for (Pair<String, String> pair : pairs) {
            Map<String, String> prop = new ConcurrentHashMap<>();
            prop.put("type", pair.getValue());
            properties.put(pair.getKey(), prop);
        }

        parameters.put("type", "object");
        parameters.put("properties", properties);
        parameters.put("required", pairs.keys());

        return builder().name(method.getName()).parameters(parameters);
    }

    /** The builder for {@code Function}. */
    public static final class Builder {

        String name;
        String description;
        Object parameters;

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder parameters(Object parameters) {
            this.parameters = parameters;
            return this;
        }

        public Builder parameters(String parameters) {
            this.parameters = JsonUtils.GSON.fromJson(parameters, JsonObject.class);
            return this;
        }

        public Function build() {
            return new Function(this);
        }
    }
}

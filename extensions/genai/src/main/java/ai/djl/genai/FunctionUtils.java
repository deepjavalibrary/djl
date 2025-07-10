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
package ai.djl.genai;

import ai.djl.util.JsonUtils;
import ai.djl.util.PairList;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** A utility class for function calling. */
public final class FunctionUtils {

    private static final Map<String, String> TYPE_MAPPING = new ConcurrentHashMap<>();

    static {
        TYPE_MAPPING.put("java.lang.String", "string");
        TYPE_MAPPING.put("boolean", "boolean");
        TYPE_MAPPING.put("Boolean", "boolean");
        TYPE_MAPPING.put("int", "integer");
        TYPE_MAPPING.put("long", "integer");
        TYPE_MAPPING.put("java.lang.Integer", "integer");
        TYPE_MAPPING.put("java.lang.Long", "integer");
        TYPE_MAPPING.put("float", "number");
        TYPE_MAPPING.put("java.lang.Float", "number");
        TYPE_MAPPING.put("double", "number");
        TYPE_MAPPING.put("java.lang.Double", "number");
    }

    private FunctionUtils() {}

    /**
     * Invokes the underlying method represented by the {@code Method} object.
     *
     * @param method the method object
     * @param obj the object the underlying method is invoked from
     * @param arguments the arguments used for the method call
     * @return the object returned by the method
     * @throws IllegalAccessException if this {@code Method} object is enforcing Java language
     *     access control and the underlying method is inaccessible.
     * @throws IllegalArgumentException if the method is not accessible
     */
    public static Object invoke(Method method, Object obj, String arguments)
            throws InvocationTargetException, IllegalAccessException {
        Map<String, String> types = FunctionUtils.getParameters(method, false).toMap();
        JsonObject args = JsonUtils.GSON.fromJson(arguments, JsonObject.class);
        List<Object> values = new ArrayList<>();
        for (Map.Entry<String, JsonElement> entry : args.entrySet()) {
            String type = types.get(entry.getKey());
            String value = entry.getValue().getAsString();
            if ("java.lang.String".equals(type)) {
                values.add(value);
            } else if ("int".equals(type) || "java.lang.Integer".equals(type)) {
                values.add(Integer.valueOf(value));
            } else if ("long".equals(type) || "java.lang.Long".equals(type)) {
                values.add(Long.valueOf(value));
            } else if ("float".equals(type) || "java.lang.Float".equals(type)) {
                values.add(Float.valueOf(value));
            } else if ("double".equals(type) || "java.lang.Double".equals(type)) {
                values.add(Double.valueOf(value));
            } else {
                throw new IllegalArgumentException("Unsupported parameter type " + type);
            }
        }
        return method.invoke(obj, values.toArray());
    }

    /**
     * Returns the method's parameter names and types.
     *
     * @param method the method
     * @return the method's parameter names and types
     */
    public static PairList<String, String> getParameters(Method method) {
        return getParameters(method, true);
    }

    private static PairList<String, String> getParameters(Method method, boolean mapping) {
        PairList<String, String> list = new PairList<>();
        Parameter[] parameters = method.getParameters();
        for (Parameter parameter : parameters) {
            if (!parameter.isNamePresent()) {
                throw new IllegalArgumentException(
                        "Failed to retrieve the parameter name from reflection. Please compile your"
                                + " code with the \"-parameters\" flag or provide parameter names"
                                + " manually.");
            }
            String parameterName = parameter.getName();
            String typeName = parameter.getType().getName();
            String type = mapping ? TYPE_MAPPING.get(typeName) : typeName;
            if (type == null) {
                throw new IllegalArgumentException(
                        "Unsupported parameter type "
                                + typeName
                                + " for parameter "
                                + parameterName
                                + ". Currently, only String and Java primitive types are"
                                + " supported");
            }
            list.add(parameterName, type);
        }
        return list;
    }
}

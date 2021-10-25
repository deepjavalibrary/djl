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
package ai.djl.translate;

import java.util.Map;

/** A utility class to extract data from model's arguments. */
public final class ArgumentsUtil {

    private ArgumentsUtil() {}

    /**
     * Returns the string value from the arguments.
     *
     * @param arguments the arguments to retrieve data
     * @param key the key to retrieve
     * @return the string value form the arguments
     */
    public static String stringValue(Map<String, ?> arguments, String key) {
        return stringValue(arguments, key, null);
    }

    /**
     * Returns the string value from the arguments.
     *
     * @param arguments the arguments to retrieve data
     * @param key the key to retrieve
     * @param def the default value if key is absent
     * @return the string value form the arguments
     */
    public static String stringValue(Map<String, ?> arguments, String key, String def) {
        Object value = arguments.get(key);
        if (value == null) {
            return def;
        }
        return value.toString();
    }

    /**
     * Returns the integer value from the arguments.
     *
     * @param arguments the arguments to retrieve data
     * @param key the key to retrieve
     * @return the integer value form the arguments
     */
    public static int intValue(Map<String, ?> arguments, String key) {
        return intValue(arguments, key, 0);
    }

    /**
     * Returns the integer value from the arguments.
     *
     * @param arguments the arguments to retrieve data
     * @param key the key to retrieve
     * @param def the default value if key is absent
     * @return the integer value form the arguments
     */
    public static int intValue(Map<String, ?> arguments, String key, int def) {
        Object value = arguments.get(key);
        if (value == null) {
            return def;
        }
        return (int) Double.parseDouble(value.toString());
    }

    /**
     * Returns the long value from the arguments.
     *
     * @param arguments the arguments to retrieve data
     * @param key the key to retrieve
     * @return the long value form the arguments
     */
    public static long longValue(Map<String, ?> arguments, String key) {
        return longValue(arguments, key, 0);
    }

    /**
     * Returns the long value from the arguments.
     *
     * @param arguments the arguments to retrieve data
     * @param key the key to retrieve
     * @param def the default value if key is absent
     * @return the long value form the arguments
     */
    public static Long longValue(Map<String, ?> arguments, String key, long def) {
        Object value = arguments.get(key);
        if (value == null) {
            return def;
        }
        return (long) Double.parseDouble(value.toString());
    }

    /**
     * Returns the float value from the arguments.
     *
     * @param arguments the arguments to retrieve data
     * @param key the key to retrieve
     * @return the float value form the arguments
     */
    public static float floatValue(Map<String, ?> arguments, String key) {
        return floatValue(arguments, key, 0f);
    }

    /**
     * Returns the float value from the arguments.
     *
     * @param arguments the arguments to retrieve data
     * @param key the key to retrieve
     * @param def the default value if key is absent
     * @return the float value form the arguments
     */
    public static float floatValue(Map<String, ?> arguments, String key, float def) {
        Object value = arguments.get(key);
        if (value == null) {
            return def;
        }
        return (float) Double.parseDouble(value.toString());
    }

    /**
     * Returns the boolean value from the arguments.
     *
     * @param arguments the arguments to retrieve data
     * @param key the key to retrieve
     * @return the boolean value form the arguments
     */
    public static boolean booleanValue(Map<String, ?> arguments, String key) {
        return booleanValue(arguments, key, false);
    }

    /**
     * Returns the boolean value from the arguments.
     *
     * @param arguments the arguments to retrieve data
     * @param key the key to retrieve
     * @param def the default value if key is absent
     * @return the boolean value form the arguments
     */
    public static boolean booleanValue(Map<String, ?> arguments, String key, boolean def) {
        Object value = arguments.get(key);
        if (value == null) {
            return def;
        }
        return Boolean.parseBoolean(value.toString());
    }
}

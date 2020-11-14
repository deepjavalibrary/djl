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
package ai.djl.paddlepaddle.engine;

import ai.djl.ndarray.types.DataType;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Helper to convert between {@link DataType} an the PaddlePaddle internal DataTypes. */
public final class PpDataType {

    private static Map<DataType, String> toPaddlePaddleMap = createMapToPaddlePaddle();
    private static Map<String, DataType> fromPaddlePaddleMap = createMapFromPaddlePaddle();

    private PpDataType() {}

    private static Map<DataType, String> createMapToPaddlePaddle() {
        Map<DataType, String> map = new ConcurrentHashMap<>();
        map.put(DataType.FLOAT32, "float32");
        map.put(DataType.FLOAT64, "float64");
        map.put(DataType.INT8, "int8");
        map.put(DataType.INT32, "int32");
        map.put(DataType.INT64, "int64");
        map.put(DataType.UINT8, "uint8");
        return map;
    }

    private static Map<String, DataType> createMapFromPaddlePaddle() {
        Map<String, DataType> map = new ConcurrentHashMap<>();
        map.put("float32", DataType.FLOAT32);
        map.put("float64", DataType.FLOAT64);
        map.put("int8", DataType.INT8);
        map.put("int32", DataType.INT32);
        map.put("int64", DataType.INT64);
        map.put("uint8", DataType.UINT8);
        return map;
    }

    /**
     * Converts a PaddlePaddle type String into a {@link DataType}.
     *
     * @param ppType the type String to convert
     * @return the {@link DataType}
     */
    public static DataType fromPaddlePaddle(String ppType) {
        return fromPaddlePaddleMap.get(ppType);
    }

    /**
     * Converts a {@link DataType} into the corresponding PaddlePaddle type String.
     *
     * @param jType the java {@link DataType} to convert
     * @return the converted PaddlePaddle type string
     */
    public static String toPaddlePaddle(DataType jType) {
        return toPaddlePaddleMap.get(jType);
    }
}

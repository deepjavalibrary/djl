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

    private static Map<DataType, Integer> toPaddlePaddleMap = createMapToPaddlePaddle();
    private static Map<Integer, DataType> fromPaddlePaddleMap = createMapFromPaddlePaddle();

    private PpDataType() {}

    private static Map<DataType, Integer> createMapToPaddlePaddle() {
        Map<DataType, Integer> map = new ConcurrentHashMap<>();
        map.put(DataType.FLOAT32, 0);
        map.put(DataType.INT64, 1);
        map.put(DataType.INT32, 2);
        map.put(DataType.INT8, 3);
        map.put(DataType.UINT8, 3);
        map.put(DataType.UNKNOWN, 4);
        return map;
    }

    private static Map<Integer, DataType> createMapFromPaddlePaddle() {
        Map<Integer, DataType> map = new ConcurrentHashMap<>();
        map.put(0, DataType.FLOAT32);
        map.put(1, DataType.INT64);
        map.put(2, DataType.INT32);
        map.put(3, DataType.INT8);
        map.put(4, DataType.UNKNOWN);
        return map;
    }

    /**
     * Converts a PaddlePaddle type String into a {@link DataType}.
     *
     * @param ppType the type String to convert
     * @return the {@link DataType}
     */
    public static DataType fromPaddlePaddle(int ppType) {
        return fromPaddlePaddleMap.get(ppType);
    }

    /**
     * Converts a {@link DataType} into the corresponding PaddlePaddle type String.
     *
     * @param jType the java {@link DataType} to convert
     * @return the converted PaddlePaddle type string
     */
    public static int toPaddlePaddle(DataType jType) {
        Integer ppType = toPaddlePaddleMap.get(jType);
        if (ppType == null) {
            throw new UnsupportedOperationException(
                    "PaddlePaddle doesn't support dataType: " + jType);
        }
        return ppType;
    }
}

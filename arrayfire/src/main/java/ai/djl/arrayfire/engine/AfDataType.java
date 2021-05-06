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
package ai.djl.arrayfire.engine;

import ai.djl.ndarray.types.DataType;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Helper to convert between {@link DataType} an the ArrayFire internal DataTypes. */
public final class AfDataType {

    private static Map<DataType, Integer> toArrayFireMap = createMapToArrayFire();
    private static Map<Integer, DataType> fromArrayFireMap = createMapFromArrayFire();

    private AfDataType() {}

    private static Map<DataType, Integer> createMapToArrayFire() {
        Map<DataType, Integer> map = new ConcurrentHashMap<>();
        map.put(DataType.FLOAT16, 12);
        map.put(DataType.FLOAT32, 0);
        map.put(DataType.FLOAT64, 2);
        map.put(DataType.BOOLEAN, 4);
        map.put(DataType.INT32, 5);
        map.put(DataType.INT64, 8);
        map.put(DataType.UINT8, 7);
        return map;
    }

    private static Map<Integer, DataType> createMapFromArrayFire() {
        Map<Integer, DataType> map = new ConcurrentHashMap<>();
        map.put(12, DataType.FLOAT16);
        map.put(0, DataType.FLOAT32);
        map.put(2, DataType.FLOAT64);
        map.put(4, DataType.BOOLEAN);
        map.put(5, DataType.INT32);
        map.put(8, DataType.INT64);
        map.put(7, DataType.UINT8);
        return map;
    }

    /**
     * Converts a ArrayFire type String into a {@link DataType}.
     *
     * @param afType the type String to convert
     * @return the {@link DataType}
     */
    public static DataType fromArrayFire(int afType) {
        return fromArrayFireMap.get(afType);
    }

    /**
     * Converts a {@link DataType} into the corresponding ArrayFire type String.
     *
     * @param jType the java {@link DataType} to convert
     * @return the converted PaddlePaddle type string
     */
    public static int toArrayFire(DataType jType) {
        Integer afType = toArrayFireMap.get(jType);
        if (afType == null) {
            throw new UnsupportedOperationException("ArrayFire doesn't support dataType: " + jType);
        }
        return afType;
    }
}

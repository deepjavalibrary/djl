/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.tensorflow.engine;

import ai.djl.ndarray.types.DataType;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public final class TfDataType {

    private static Map<DataType, Integer> toTfMap = createToTfMap();
    private static Map<Integer, DataType> fromTfMap = createFromTfMap();

    private TfDataType() {}

    private static Map<DataType, Integer> createToTfMap() {
        Map<DataType, Integer> map = new ConcurrentHashMap<>();
        map.put(DataType.FLOAT32, 1);
        map.put(DataType.FLOAT64, 2);
        map.put(DataType.INT32, 3);
        map.put(DataType.INT64, 9);
        map.put(DataType.UINT8, 4);
        map.put(DataType.INT8, 6);
        map.put(DataType.BOOLEAN, 10);
        map.put(DataType.STRING, 7);
        return map;
    }

    private static Map<Integer, DataType> createFromTfMap() {
        Map<Integer, DataType> map = new ConcurrentHashMap<>();
        map.put(1, DataType.FLOAT32);
        map.put(2, DataType.FLOAT64);
        map.put(3, DataType.INT32);
        map.put(4, DataType.UINT8);
        map.put(6, DataType.INT8);
        map.put(7, DataType.STRING);
        map.put(9, DataType.INT64);
        map.put(10, DataType.BOOLEAN);
        return map;
    }

    public static int toTf(DataType dataType) {
        return toTfMap.get(dataType);
    }

    public static DataType fromTf(int dataType) {
        return fromTfMap.get(dataType);
    }
}

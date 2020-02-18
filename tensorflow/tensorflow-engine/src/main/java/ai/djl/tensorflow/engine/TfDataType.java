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

    private static Map<DataType, org.tensorflow.DataType> toTf = createMapToTf();
    private static Map<org.tensorflow.DataType, DataType> fromTf = createMapFromTf();

    private TfDataType() {}

    private static Map<DataType, org.tensorflow.DataType> createMapToTf() {
        Map<DataType, org.tensorflow.DataType> map = new ConcurrentHashMap<>();
        map.put(DataType.FLOAT32, org.tensorflow.DataType.FLOAT);
        map.put(DataType.FLOAT64, org.tensorflow.DataType.DOUBLE);
        map.put(DataType.INT32, org.tensorflow.DataType.INT32);
        map.put(DataType.INT64, org.tensorflow.DataType.INT64);
        map.put(DataType.UINT8, org.tensorflow.DataType.UINT8);
        return map;
    }

    private static Map<org.tensorflow.DataType, DataType> createMapFromTf() {
        Map<org.tensorflow.DataType, DataType> map = new ConcurrentHashMap<>();
        map.put(org.tensorflow.DataType.FLOAT, DataType.FLOAT32);
        map.put(org.tensorflow.DataType.DOUBLE, DataType.FLOAT64);
        map.put(org.tensorflow.DataType.INT32, DataType.INT32);
        map.put(org.tensorflow.DataType.INT64, DataType.INT64);
        map.put(org.tensorflow.DataType.UINT8, DataType.UINT8);
        return map;
    }

    public static DataType fromTf(org.tensorflow.DataType tfType) {
        return fromTf.get(tfType);
    }

    public static org.tensorflow.DataType toTf(DataType jType) {
        return toTf.get(jType);
    }
}

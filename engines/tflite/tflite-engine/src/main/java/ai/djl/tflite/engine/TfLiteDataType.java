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
package ai.djl.tflite.engine;

import ai.djl.ndarray.types.DataType;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Converts between DJL and TFLite data types. */
public final class TfLiteDataType {

    private static Map<DataType, org.tensorflow.lite.DataType> toTf = createMapToTf();
    private static Map<org.tensorflow.lite.DataType, DataType> fromTf = createMapFromTf();

    private TfLiteDataType() {}

    private static Map<DataType, org.tensorflow.lite.DataType> createMapToTf() {
        Map<DataType, org.tensorflow.lite.DataType> map = new ConcurrentHashMap<>();
        map.put(DataType.FLOAT32, org.tensorflow.lite.DataType.FLOAT32);
        map.put(DataType.INT32, org.tensorflow.lite.DataType.INT32);
        map.put(DataType.INT64, org.tensorflow.lite.DataType.INT64);
        map.put(DataType.UINT8, org.tensorflow.lite.DataType.UINT8);
        map.put(DataType.INT8, org.tensorflow.lite.DataType.INT8);
        map.put(DataType.BOOLEAN, org.tensorflow.lite.DataType.BOOL);
        map.put(DataType.STRING, org.tensorflow.lite.DataType.STRING);
        return map;
    }

    private static Map<org.tensorflow.lite.DataType, DataType> createMapFromTf() {
        Map<org.tensorflow.lite.DataType, DataType> map = new ConcurrentHashMap<>();
        map.put(org.tensorflow.lite.DataType.FLOAT32, DataType.FLOAT32);
        map.put(org.tensorflow.lite.DataType.INT32, DataType.INT32);
        map.put(org.tensorflow.lite.DataType.INT64, DataType.INT64);
        map.put(org.tensorflow.lite.DataType.UINT8, DataType.UINT8);
        map.put(org.tensorflow.lite.DataType.BOOL, DataType.BOOLEAN);
        map.put(org.tensorflow.lite.DataType.STRING, DataType.STRING);
        return map;
    }

    /**
     * Converts from a TFLite data type to a DJL data type.
     *
     * @param tfType the TFLite data type
     * @return the DJL data type
     */
    public static DataType fromTf(org.tensorflow.lite.DataType tfType) {
        return fromTf.get(tfType);
    }

    /**
     * Converts from a DJL data type to a TFLite data type.
     *
     * @param type the DJL data type
     * @return the TFLite data type
     */
    public static org.tensorflow.lite.DataType toTf(DataType type) {
        return toTf.get(type);
    }
}

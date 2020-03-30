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
import org.tensorflow.types.TBool;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TUint8;
import org.tensorflow.types.family.TType;

public final class TfDataType {

    private static Map<DataType, org.tensorflow.DataType<? extends TType>> toTf = createMapToTf();
    private static Map<org.tensorflow.DataType<? extends TType>, DataType> fromTf =
            createMapFromTf();

    private TfDataType() {}

    private static Map<DataType, org.tensorflow.DataType<? extends TType>> createMapToTf() {
        Map<DataType, org.tensorflow.DataType<? extends TType>> map = new ConcurrentHashMap<>();
        map.put(DataType.FLOAT32, TFloat32.DTYPE);
        map.put(DataType.FLOAT64, TFloat64.DTYPE);
        map.put(DataType.INT32, TInt32.DTYPE);
        map.put(DataType.INT64, TInt64.DTYPE);
        map.put(DataType.UINT8, TUint8.DTYPE);
        map.put(DataType.BOOLEAN, TBool.DTYPE);
        return map;
    }

    private static Map<org.tensorflow.DataType<? extends TType>, DataType> createMapFromTf() {
        Map<org.tensorflow.DataType<? extends TType>, DataType> map = new ConcurrentHashMap<>();
        map.put(TFloat32.DTYPE, DataType.FLOAT32);
        map.put(TFloat64.DTYPE, DataType.FLOAT64);
        map.put(TInt32.DTYPE, DataType.INT32);
        map.put(TInt64.DTYPE, DataType.INT64);
        map.put(TUint8.DTYPE, DataType.UINT8);
        map.put(TBool.DTYPE, DataType.BOOLEAN);
        return map;
    }

    public static DataType fromTf(org.tensorflow.DataType<? extends TType> tfType) {
        return fromTf.get(tfType);
    }

    public static org.tensorflow.DataType<? extends TType> toTf(DataType type) {
        return toTf.get(type);
    }
}

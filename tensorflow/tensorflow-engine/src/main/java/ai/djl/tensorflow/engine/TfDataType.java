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
import org.tensorflow.types.TString;
import org.tensorflow.types.TUint8;
import org.tensorflow.types.family.TType;

public final class TfDataType {

    private static Map<DataType, Class<? extends TType>> classTypeTo = createClassTypeTo();
    private static Map<org.tensorflow.proto.framework.DataType, DataType> protoTypeFrom =
            createProtoTypeFrom();
    private static Map<DataType, org.tensorflow.proto.framework.DataType> protoTypeTo =
            createProtoTypeTo();

    private TfDataType() {}

    private static Map<DataType, Class<? extends TType>> createClassTypeTo() {
        Map<DataType, Class<? extends TType>> map = new ConcurrentHashMap<>();
        map.put(DataType.FLOAT32, TFloat32.class);
        map.put(DataType.FLOAT64, TFloat64.class);
        map.put(DataType.INT32, TInt32.class);
        map.put(DataType.INT64, TInt64.class);
        map.put(DataType.UINT8, TUint8.class);
        map.put(DataType.INT8, TUint8.class);
        map.put(DataType.BOOLEAN, TBool.class);
        map.put(DataType.STRING, TString.class);
        return map;
    }

    private static Map<org.tensorflow.proto.framework.DataType, DataType> createProtoTypeFrom() {
        Map<org.tensorflow.proto.framework.DataType, DataType> map = new ConcurrentHashMap<>();
        map.put(org.tensorflow.proto.framework.DataType.DT_FLOAT, DataType.FLOAT32);
        map.put(org.tensorflow.proto.framework.DataType.DT_DOUBLE, DataType.FLOAT64);
        map.put(org.tensorflow.proto.framework.DataType.DT_INT32, DataType.INT32);
        map.put(org.tensorflow.proto.framework.DataType.DT_INT64, DataType.INT64);
        map.put(org.tensorflow.proto.framework.DataType.DT_UINT8, DataType.UINT8);
        map.put(org.tensorflow.proto.framework.DataType.DT_INT8, DataType.INT8);
        map.put(org.tensorflow.proto.framework.DataType.DT_BOOL, DataType.BOOLEAN);
        map.put(org.tensorflow.proto.framework.DataType.DT_STRING, DataType.STRING);
        return map;
    }

    private static Map<DataType, org.tensorflow.proto.framework.DataType> createProtoTypeTo() {
        Map<DataType, org.tensorflow.proto.framework.DataType> map = new ConcurrentHashMap<>();
        map.put(DataType.FLOAT32, org.tensorflow.proto.framework.DataType.DT_FLOAT);
        map.put(DataType.FLOAT64, org.tensorflow.proto.framework.DataType.DT_DOUBLE);
        map.put(DataType.INT32, org.tensorflow.proto.framework.DataType.DT_INT32);
        map.put(DataType.INT64, org.tensorflow.proto.framework.DataType.DT_INT64);
        map.put(DataType.UINT8, org.tensorflow.proto.framework.DataType.DT_UINT8);
        map.put(DataType.INT8, org.tensorflow.proto.framework.DataType.DT_UINT8);
        map.put(DataType.BOOLEAN, org.tensorflow.proto.framework.DataType.DT_BOOL);
        map.put(DataType.STRING, org.tensorflow.proto.framework.DataType.DT_STRING);
        return map;
    }

    public static Class<? extends TType> toClassType(DataType type) {
        return classTypeTo.get(type);
    }

    public static DataType fromProtoType(org.tensorflow.proto.framework.DataType tfType) {
        return protoTypeFrom.get(tfType);
    }

    public static org.tensorflow.proto.framework.DataType toProtoType(DataType dataType) {
        return protoTypeTo.get(dataType);
    }
}

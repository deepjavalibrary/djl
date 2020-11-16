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
package ai.djl.paddlepaddle.jna;

import ai.djl.ndarray.types.DataType;
import ai.djl.paddlepaddle.engine.PpDataType;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

/**
 * A class containing utilities to interact with the MXNet Engine's Java Native Access (JNA) layer.
 */
@SuppressWarnings("MissingJavadocMethod")
public final class JnaUtils {

    private static final PaddleLibrary LIB = Native.load(PaddleLibrary.class);

    private JnaUtils() {}

    public static DataType getDataType(Pointer pointer) {
        int type = LIB.PD_GetPaddleTensorDType(pointer);
        return PpDataType.fromPaddlePaddle(type);
    }
}

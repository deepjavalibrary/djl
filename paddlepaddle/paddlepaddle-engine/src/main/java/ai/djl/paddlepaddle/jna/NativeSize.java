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

import com.sun.jna.IntegerType;
import com.sun.jna.Native;

/** The Paddle Native size handler. */
@SuppressWarnings("missingjavadocmethod")
public class NativeSize extends IntegerType {

    private static final long serialVersionUID = 1L;

    public static final int SIZE = Native.SIZE_T_SIZE;

    public NativeSize() {
        this(0);
    }

    public NativeSize(long value) {
        super(SIZE, value);
    }
}

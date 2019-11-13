/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.mxnet.test;

import com.sun.jna.Native;
import com.sun.jna.Pointer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

public final class TestHelper {

    private TestHelper() {}

    public static Pointer toPointer(String val) {
        byte[] buf = val.getBytes(StandardCharsets.UTF_8);
        byte[] dest = new byte[buf.length + 1];
        System.arraycopy(buf, 0, dest, 0, buf.length);
        return toPointer(dest);
    }

    public static Pointer toPointer(int[] arr) {
        ByteBuffer bb = ByteBuffer.allocateDirect(arr.length * 4);
        bb.order(ByteOrder.nativeOrder());
        bb.asIntBuffer().put(arr);
        bb.rewind();
        return Native.getDirectBufferPointer(bb);
    }

    public static Pointer toPointer(byte[] buf) {
        ByteBuffer bb = ByteBuffer.allocateDirect(buf.length);
        bb.order(ByteOrder.nativeOrder());
        bb.put(buf);
        bb.rewind();
        return Native.getDirectBufferPointer(bb);
    }
}

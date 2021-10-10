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
package ai.djl.ndarray;

import ai.djl.util.JsonUtils;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

class BytesSupplierImpl implements BytesSupplier {

    private byte[] buf;
    private String value;
    private Object obj;

    BytesSupplierImpl(byte[] buf) {
        this.buf = buf;
    }

    BytesSupplierImpl(String value) {
        this.value = value;
    }

    BytesSupplierImpl(Object obj) {
        this.obj = obj;
    }

    /** {@inheritDoc} */
    @Override
    public byte[] getAsBytes() {
        if (buf == null) {
            if (value == null) {
                value = JsonUtils.GSON_PRETTY.toJson(obj) + '\n';
            }
            buf = value.getBytes(StandardCharsets.UTF_8);
        }
        return buf;
    }

    /** {@inheritDoc} */
    @Override
    public String getAsString() {
        if (value == null) {
            if (obj != null) {
                value = JsonUtils.GSON_PRETTY.toJson(obj) + '\n';
            } else {
                value = new String(buf, StandardCharsets.UTF_8);
            }
        }
        return value;
    }

    /** {@inheritDoc} */
    @Override
    public Object getAsObject() {
        if (obj != null) {
            return obj;
        } else if (value != null) {
            return value;
        }
        return buf;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        return ByteBuffer.wrap(getAsBytes());
    }
}

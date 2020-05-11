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

package ai.djl.ndarray;

import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

/** A class contains encoding and decoding logic for NDArray. */
final class NDSerializer {

    static final int BUFFER_SIZE = 81920;
    static final String MAGIC_NUMBER = "NDAR";
    static final int VERSION = 1;

    private NDSerializer() {}

    /**
     * Encodes {@link NDArray} to byte array.
     *
     * @param array the input {@link NDArray}
     * @return byte array
     */
    static byte[] encode(NDArray array) {
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            DataOutputStream dos = new DataOutputStream(baos);
            // magic string for version identification
            dos.writeUTF(MAGIC_NUMBER);
            dos.writeInt(VERSION);
            dos.writeUTF(array.getSparseFormat().name());
            dos.writeUTF(array.getDataType().name());

            Shape shape = array.getShape();
            dos.write(shape.getEncoded());

            ByteBuffer bb = array.toByteBuffer();
            int length = bb.remaining();
            dos.writeInt(length);

            if (length > 0) {
                if (length > BUFFER_SIZE) {
                    byte[] buf = new byte[BUFFER_SIZE];
                    while (length > BUFFER_SIZE) {
                        bb.get(buf);
                        dos.write(buf);
                        length = bb.remaining();
                    }
                }

                byte[] buf = new byte[length];
                bb.get(buf);
                dos.write(buf);
            }
            dos.flush();
            return baos.toByteArray();
        } catch (IOException e) {
            throw new AssertionError("This should never happen", e);
        }
    }

    /**
     * Decodes {@link NDArray} through {@link DataInputStream}.
     *
     * @param manager the {@link NDManager} assigned to the {@link NDArray}
     * @param is input stream data to load from
     * @return {@link NDArray}
     * @throws IOException data is not readable
     */
    static NDArray decode(NDManager manager, InputStream is) throws IOException {
        DataInputStream dis;
        if (is instanceof DataInputStream) {
            dis = (DataInputStream) is;
        } else {
            dis = new DataInputStream(is);
        }

        // Newer version of NDArray
        if ("NDAR".equals(dis.readUTF())) {
            // NDArray encode version
            int version = dis.readInt();
            if (version != 1) {
                throw new IllegalArgumentException("Unexpected NDArray encode version " + version);
            }
            dis.readUTF(); // ignore SparseFormat
        }
        // else ignored as reading SparseFormat for the old version

        // DataType - 1 byte
        DataType dataType = DataType.valueOf(dis.readUTF());

        // Shape
        Shape shape = Shape.decode(dis);

        // Data
        int length = dis.readInt();
        ByteBuffer data = manager.allocateDirect(length);

        if (length > 0) {
            byte[] buf = new byte[BUFFER_SIZE];
            while (length > BUFFER_SIZE) {
                dis.readFully(buf);
                data.put(buf);
                length -= BUFFER_SIZE;
            }

            dis.readFully(buf, 0, length);
            data.put(buf, 0, length);
            data.rewind();
        }
        return manager.create(dataType.asDataType(data), shape);
    }
}

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
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** A class contains encoding and decoding logic for NDArray. */
final class NDSerializer {

    private static final int VERSION = 3;

    private static final int BUFFER_SIZE = 1024 * 1024;
    private static final String MAGIC_NUMBER = "NDAR";
    private static final byte[] NUMPY_MAGIC = {(byte) 0x93, 'N', 'U', 'M', 'P', 'Y'};
    private static final int ARRAY_ALIGN = 64;

    private static final Pattern PATTERN =
            Pattern.compile("\\{'descr': '(.+)', 'fortran_order': False, 'shape': \\((.*)\\),");

    private NDSerializer() {}

    /**
     * Encodes {@link NDArray} to byte array.
     *
     * @param array the input {@link NDArray}
     * @return byte array
     */
    static byte[] encode(NDArray array) {
        int total = Math.toIntExact(array.size()) * array.getDataType().getNumOfBytes() + 100;
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream(total)) {
            encode(array, baos);
            return baos.toByteArray();
        } catch (IOException e) {
            throw new AssertionError("This should never happen", e);
        }
    }

    static void encode(NDArray array, OutputStream os) throws IOException {
        DataOutputStream dos;
        if (os instanceof DataOutputStream) {
            dos = (DataOutputStream) os;
        } else {
            dos = new DataOutputStream(os);
        }
        // magic string for version identification
        dos.writeUTF(MAGIC_NUMBER);
        dos.writeInt(VERSION);
        String name = array.getName();
        if (name == null) {
            dos.write(0);
        } else {
            dos.write(1);
            dos.writeUTF(name);
        }
        dos.writeUTF(array.getSparseFormat().name());
        dos.writeUTF(array.getDataType().name());

        Shape shape = array.getShape();
        dos.write(shape.getEncoded());

        if (array.getDataType() == DataType.STRING) {
            String[] data = array.toStringArray();
            dos.writeInt(data.length);
            for (String str : data) {
                dos.writeUTF(str);
            }
            dos.flush();
            return;
        }

        ByteBuffer bb = array.toByteBuffer();
        dos.write(bb.order() == ByteOrder.BIG_ENDIAN ? '>' : '<');
        int length = bb.remaining();
        dos.writeInt(length);

        if (length > 0) {
            if (bb.hasArray() && bb.remaining() == bb.array().length) {
                dos.write(bb.array(), bb.position(), length);
            } else {
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
        }
        dos.flush();
    }

    static void encodeAsNumpy(NDArray array, OutputStream os) throws IOException {
        StringBuilder sb = new StringBuilder(80);
        sb.append("{'descr': '")
                .append(array.getDataType().asNumpy())
                .append("', 'fortran_order': False, 'shape': ");
        long[] shape = array.getShape().getShape();
        if (shape.length == 1) {
            sb.append('(').append(shape[0]).append(",)");
        } else {
            sb.append(array.getShape());
        }
        sb.append(", }");

        int len = sb.length() + 1;
        int padding = ARRAY_ALIGN - (NUMPY_MAGIC.length + len + 4) % ARRAY_ALIGN;
        ByteBuffer bb = ByteBuffer.allocate(2);
        bb.order(ByteOrder.LITTLE_ENDIAN);
        bb.putShort((short) (padding + len));

        os.write(NUMPY_MAGIC);
        os.write(1);
        os.write(0); // version 1.0
        os.write(bb.array());
        os.write(sb.toString().getBytes(StandardCharsets.US_ASCII));
        for (int i = 0; i < padding; ++i) {
            os.write(' ');
        }
        os.write('\n');
        os.write(array.toByteArray());
    }

    static NDArray decode(NDManager manager, ByteBuffer bb) {
        if (!"NDAR".equals(readUTF(bb))) {
            throw new IllegalArgumentException("Malformed NDArray data");
        }

        // NDArray encode version
        int version = bb.getInt();
        if (version < 1 || version > VERSION) {
            throw new IllegalArgumentException("Unexpected NDArray encode version " + version);
        }

        String name = null;
        if (version > 1) {
            byte flag = bb.get();
            if (flag == 1) {
                name = readUTF(bb);
            }
        }

        readUTF(bb); // ignore SparseFormat

        // DataType
        DataType dataType = DataType.valueOf(readUTF(bb));

        // Shape
        Shape shape = Shape.decode(bb);

        if (dataType == DataType.STRING) {
            int size = bb.getInt();
            String[] data = new String[size];
            for (int i = 0; i < size; ++i) {
                data[i] = readUTF(bb);
            }
            NDArray array = manager.create(data, StandardCharsets.UTF_8, shape);
            array.setName(name);
            return array;
        }

        // Data
        ByteOrder order;
        if (version > 2) {
            order = bb.get() == '>' ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN;
        } else {
            order = ByteOrder.nativeOrder();
        }
        int length = bb.getInt();
        ByteBuffer data = bb.slice();
        data.limit(length);
        data.order(order);

        NDArray array = manager.create(data, shape, dataType);
        array.setName(name);
        bb.position(bb.position() + length);
        return array;
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

        if (!"NDAR".equals(dis.readUTF())) {
            throw new IllegalArgumentException("Malformed NDArray data");
        }

        // NDArray encode version
        int version = dis.readInt();
        if (version < 1 || version > VERSION) {
            throw new IllegalArgumentException("Unexpected NDArray encode version " + version);
        }

        String name = null;
        if (version > 1) {
            byte flag = dis.readByte();
            if (flag == 1) {
                name = dis.readUTF();
            }
        }

        dis.readUTF(); // ignore SparseFormat

        // DataType
        DataType dataType = DataType.valueOf(dis.readUTF());

        // Shape
        Shape shape = Shape.decode(dis);

        // Data
        ByteOrder order;
        if (version > 2) {
            order = dis.readByte() == '>' ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN;
        } else {
            order = ByteOrder.nativeOrder();
        }
        int length = dis.readInt();
        ByteBuffer data = manager.allocateDirect(length);
        data.order(order);
        readData(dis, data, length);

        NDArray array = manager.create(data, shape, dataType);
        array.setName(name);
        return array;
    }

    static NDArray decodeNumpy(NDManager manager, InputStream is) throws IOException {
        DataInputStream dis;
        if (is instanceof DataInputStream) {
            dis = (DataInputStream) is;
        } else {
            dis = new DataInputStream(is);
        }

        byte[] buf = new byte[NUMPY_MAGIC.length];
        dis.readFully(buf);
        if (!Arrays.equals(buf, NUMPY_MAGIC)) {
            throw new IllegalArgumentException("Malformed numpy data");
        }
        byte major = dis.readByte();
        byte minor = dis.readByte();
        if (major < 1 || major > 3 || minor != 0) {
            throw new IllegalArgumentException("Unknown numpy version: " + major + '.' + minor);
        }
        int len = major == 1 ? 2 : 4;
        dis.readFully(buf, 0, len);
        ByteBuffer bb = ByteBuffer.wrap(buf, 0, len);
        bb.order(ByteOrder.LITTLE_ENDIAN);
        if (major == 1) {
            len = bb.getShort();
        } else {
            len = bb.getInt();
        }
        buf = new byte[len];
        dis.readFully(buf);
        String header = new String(buf, StandardCharsets.UTF_8).trim();
        Matcher m = PATTERN.matcher(header);
        if (!m.find()) {
            throw new IllegalArgumentException("Invalid numpy header: " + header);
        }
        String typeStr = m.group(1);
        DataType dataType = DataType.fromNumpy(typeStr);
        String shapeStr = m.group(2);
        long[] longs;
        if (shapeStr.isEmpty()) {
            longs = new long[0];
        } else {
            String[] tokens = shapeStr.split(", ?");
            longs = Arrays.stream(tokens).mapToLong(Long::parseLong).toArray();
        }
        Shape shape = new Shape(longs);
        len = Math.toIntExact(shape.size() * dataType.getNumOfBytes());
        ByteBuffer data = manager.allocateDirect(len);
        char order = typeStr.charAt(0);
        if (order == '>') {
            data.order(ByteOrder.BIG_ENDIAN);
        } else if (order == '<') {
            data.order(ByteOrder.LITTLE_ENDIAN);
        }
        readData(dis, data, len);

        return manager.create(data, shape, dataType);
    }

    private static void readData(DataInputStream dis, ByteBuffer data, int len) throws IOException {
        if (len > 0) {
            byte[] buf = new byte[BUFFER_SIZE];
            while (len > BUFFER_SIZE) {
                dis.readFully(buf);
                data.put(buf);
                len -= BUFFER_SIZE;
            }

            dis.readFully(buf, 0, len);
            data.put(buf, 0, len);
            data.rewind();
        }
    }

    private static String readUTF(ByteBuffer bb) {
        int len = (bb.get() & 0xFF << 8) + (bb.get() & 0xFF);
        byte[] buf = new byte[len];
        char[] chars = new char[len];
        bb.get(buf);

        int chararrCount = 0;
        int count = 0;
        while (count < len) {
            int c = buf[count] & 0xff;
            if (c > 127) {
                break;
            }
            count++;
            chars[chararrCount++] = (char) c;
        }

        while (count < len) {
            int c = buf[count] & 0xff;
            switch (c >> 4) {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                    /* 0xxxxxxx */
                    count++;
                    chars[chararrCount++] = (char) c;
                    break;
                case 12:
                case 13:
                    // 110x xxxx 10xx xxxx
                    count += 2;
                    if (count > len) {
                        throw new IllegalArgumentException("malformed UTF-8 input");
                    }
                    int char2 = buf[count - 1];
                    if ((char2 & 0xC0) != 0x80) {
                        throw new IllegalArgumentException("malformed input around byte " + count);
                    }
                    chars[chararrCount++] = (char) (((c & 0x1F) << 6) | (char2 & 0x3F));
                    break;
                case 14:
                    // 1110 xxxx 10xx xxxx 10xx xxxx
                    count += 3;
                    if (count > len) {
                        throw new IllegalArgumentException("malformed UTF-8 input");
                    }
                    char2 = buf[count - 2];
                    int char3 = buf[count - 1];
                    if (((char2 & 0xC0) != 0x80) || ((char3 & 0xC0) != 0x80)) {
                        throw new IllegalArgumentException("malformed UTF-8 input");
                    }
                    chars[chararrCount++] =
                            (char) (((c & 0x0F) << 12) | ((char2 & 0x3F) << 6) | (char3 & 0x3F));
                    break;
                default:
                    // 10xx xxxx, 1111 xxxx
                    throw new IllegalArgumentException("malformed input around byte " + count);
            }
        }
        return new String(chars, 0, chararrCount);
    }
}

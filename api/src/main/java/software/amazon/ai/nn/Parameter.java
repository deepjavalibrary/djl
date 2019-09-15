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
package software.amazon.ai.nn;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Objects;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.LayoutType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.training.initializer.Initializer;

public class Parameter implements AutoCloseable {

    private static final byte VERSION = 1;

    private static final int BUFFER_SIZE = 81920;

    private NDManager manager;
    private String name;
    private Block block;
    private ParameterType type;
    private Initializer initializer;
    private NDArray array;

    public Parameter(String name, Block block, ParameterType type) {
        this(name, block, type, null);
    }

    public Parameter(String name, Block block, ParameterType type, Initializer initializer) {
        this.name = name;
        this.block = block;
        this.type = type;
        this.initializer = initializer;
    }

    public Parameter(String name, Block block, NDArray array) {
        manager = array.getManager();
        this.name = name;
        this.block = block;
        this.array = array;
    }

    public String getName() {
        return name;
    }

    public ParameterType getType() {
        return type;
    }

    public NDArray getArray() {
        if (!isInitialized()) {
            throw new IllegalStateException("The array has not been initialized");
        }
        return array;
    }

    public boolean isInitialized() {
        return array != null;
    }

    public Parameter setInitializer(NDManager manager, Initializer initializer) {
        setInitializer(manager, initializer, false);
        return this;
    }

    public Parameter setInitializer(NDManager manager, Initializer initializer, boolean overwrite) {
        this.manager = manager;
        if (overwrite || this.initializer == null) {
            this.initializer = initializer;
        }
        return this;
    }

    public void reinitialize() {
        Objects.requireNonNull(initializer, "No initializer has been set");
        if (!isInitialized()) {
            throw new IllegalStateException("This parameter is not initialized");
        }

        array = initializer.initialize(manager, array.getShape(), array.getDataType());
        array.attachGradient();

        // TODO: close old array
    }

    public void initialize(NDList inputs, boolean overwrite) {
        Objects.requireNonNull(initializer, "No initializer has been set");

        if (isInitialized()) {
            if (!overwrite) {
                throw new IllegalStateException("This parameter is already initialized");
            }
            // TODO: close old array
        }

        array =
                initializer.initialize(
                        manager,
                        block.getParameterShape(name, inputs),
                        inputs.head().getDataType());
        array.attachGradient();
    }

    public void save(DataOutputStream dos) throws IOException {
        if (array == null) {
            dos.writeChar('N');
            return;
        }

        dos.writeChar('P');
        dos.writeByte(VERSION);

        dos.writeUTF(array.getSparseFormat().name());
        dos.writeUTF(array.getDataType().name());

        Shape shape = array.getShape();
        long[] shapeValue = shape.getShape();
        dos.writeInt(shapeValue.length);
        for (long l : shapeValue) {
            dos.writeLong(l);
        }
        LayoutType[] layout = shape.getLayout();
        dos.writeInt(layout.length);
        for (LayoutType layoutType : layout) {
            dos.writeChar(layoutType.getValue());
        }

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
    }

    /**
     * Load parameter ndarrays from InputStream.
     *
     * <p>Currently, we cannot deserialize into exact subclass of NDArray. The SparseNDArray and
     * Matrix will be restored as NDArray only.
     *
     * @param manager NDManager
     * @param dis InputStream
     * @throws IOException if failed to write
     */
    public void load(NDManager manager, DataInputStream dis) throws IOException {
        char magic = dis.readChar();
        if (magic == 'N') {
            return;
        } else if (magic != 'P') {
            throw new IllegalArgumentException("Invalid input data.");
        }

        // Version
        byte version = dis.readByte();
        if (version != VERSION) {
            throw new IllegalArgumentException("Unsupported encoding version: " + version);
        }

        dis.readUTF(); // ignore SparseFormat

        // DataType - 1 byte
        DataType dataType = DataType.valueOf(dis.readUTF());

        // Shape
        int length = dis.readInt();
        long[] shapeValue = new long[length];
        for (int i = 0; i < length; ++i) {
            shapeValue[i] = dis.readLong();
        }

        // Layout
        length = dis.readInt();
        char[] layout = new char[length];
        for (int i = 0; i < length; ++i) {
            layout[i] = dis.readChar();
        }
        Shape shape = new Shape(shapeValue, new String(layout));

        // Data
        length = dis.readInt();
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

        array = manager.create(dataType.asDataType(data), shape);
    }

    @Override
    public void close() {
        array.close();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Parameter parameter = (Parameter) o;
        return Objects.equals(name, parameter.name)
                && type == parameter.type
                && Objects.equals(array, parameter.array);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, type, array);
    }
}

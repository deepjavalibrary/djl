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
import software.amazon.ai.Device;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.training.ParameterStore;
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
    // use ParameterStore to store copies of array on multi devices
    private ParameterStore parameterStore;

    public Parameter(String name, Block block, ParameterType type) {
        this.name = name;
        this.block = block;
        this.type = type;
        this.initializer = type.getInitializer();
    }

    public Parameter(String name, Block block, NDArray array, ParameterType type) {
        manager = array.getManager();
        this.name = name;
        this.block = block;
        this.array = array;
        this.type = type;
        this.initializer = type.getInitializer();
    }

    public String getName() {
        return name == null ? "" : name;
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

    public NDArray getArray(Device device) {
        if (parameterStore != null) {
            return parameterStore.getValue(this, device);
        } else {
            return getArray();
        }
    }

    public boolean isInitialized() {
        return array != null;
    }

    public void setInitializer(NDManager manager, Initializer initializer, boolean overwrite) {
        this.manager = manager;
        if (overwrite || this.initializer == null) {
            this.initializer = initializer;
        }
    }

    public void initialize(NDList inputs) {

        if (!isInitialized()) {
            Objects.requireNonNull(initializer, "No initializer has been set");
            Shape[] shapes = new Shape[inputs.size()];
            for (int i = 0; i < shapes.length; ++i) {
                shapes[i] = inputs.get(i).getShape();
            }
            array =
                    initializer.initialize(
                            manager,
                            block.getParameterShape(name, shapes),
                            inputs.head().getDataType());
        }

        if (parameterStore != null) {
            parameterStore.initialize(this, array);
        } else {
            // TODO: If no initializer
            if (initializer != null) {
                array.attachGradient();
            }
        }
    }

    public void setParameterStore(ParameterStore parameterStore) {
        this.parameterStore = parameterStore;
    }

    public void save(DataOutputStream dos) throws IOException {
        if (!isInitialized()) {
            dos.writeChar('N');
            return;
        }

        dos.writeChar('P');
        dos.writeByte(VERSION);

        dos.writeUTF(getName());

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

        String parameterName = dis.readUTF();
        if (!parameterName.equals(getName())) {
            throw new IllegalArgumentException(
                    "Unexpected parameter name: " + parameterName + ", expected: " + name);
        }

        dis.readUTF(); // ignore SparseFormat

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

        array = manager.create(dataType.asDataType(data), shape);
    }

    @Override
    public void close() {
        if (array != null) {
            array.close();
            array = null;
        }
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
                && Objects.equals(array, parameter.getArray());
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, type, array);
    }
}

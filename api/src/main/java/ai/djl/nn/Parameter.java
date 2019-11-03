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
package ai.djl.nn;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.initializer.Initializer;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Objects;
import java.util.UUID;

public class Parameter implements AutoCloseable {

    private static final byte VERSION = 1;

    private static final int BUFFER_SIZE = 81920;

    private String id;
    private String name;
    private Block block;
    private ParameterType type;
    private DataType mandatoryDataType;
    private Initializer initializer;
    private NDArray array;
    private boolean requireGrad;

    public Parameter(String name, Block block, ParameterType type) {
        this(name, block, type, true);
    }

    public Parameter(String name, Block block, ParameterType type, boolean requireGrad) {
        this.id = UUID.randomUUID().toString();
        this.name = name;
        this.block = block;
        this.type = type;
        this.requireGrad = requireGrad;
        this.initializer = type.getInitializer();
    }

    public String getId() {
        return id;
    }

    public String getName() {
        return name == null ? "" : name;
    }

    public ParameterType getType() {
        return type;
    }

    public void setArray(NDArray array) {
        this.array = array;
    }

    public NDArray getArray() {
        if (!isInitialized()) {
            throw new IllegalStateException("The array has not been initialized");
        }
        return array;
    }

    public boolean requireGradient() {
        return requireGrad;
    }

    public void setMandatoryDataType(DataType mandatoryDataType) {
        this.mandatoryDataType = mandatoryDataType;
    }

    public boolean isInitialized() {
        return array != null;
    }

    public void setInitializer(Initializer initializer, boolean overwrite) {
        if (overwrite || this.initializer == null) {
            this.initializer = initializer;
        }
    }

    public void initialize(NDManager manager, DataType dataType, Shape[] inputShapes) {
        Objects.requireNonNull(initializer, "No initializer has been set");
        if (!isInitialized()) {
            Shape shape = block.getParameterShape(name, inputShapes);
            array =
                    initializer.initialize(
                            manager,
                            shape,
                            mandatoryDataType == null ? dataType : mandatoryDataType);
        }

        if (requireGradient()) {
            array.attachGradient();
        }
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
     * @throws MalformedModelException Exception thrown when model is not in expected format
     *     (parameters).
     */
    public void load(NDManager manager, DataInputStream dis)
            throws IOException, MalformedModelException {
        char magic = dis.readChar();
        if (magic == 'N') {
            return;
        } else if (magic != 'P') {
            throw new MalformedModelException("Invalid input data.");
        }

        // Version
        byte version = dis.readByte();
        if (version != VERSION) {
            throw new MalformedModelException("Unsupported encoding version: " + version);
        }

        String parameterName = dis.readUTF();
        if (!parameterName.equals(getName())) {
            throw new MalformedModelException(
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

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (array != null) {
            array.close();
            array = null;
        }
    }
}

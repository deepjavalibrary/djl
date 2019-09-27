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
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import software.amazon.ai.Device;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataType;
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
    private Map<Integer, NDArray> arrays = new ConcurrentHashMap<>();
    private Device[] devices;

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
        Device device = array.getDevice();
        devices = new Device[] {device};
        arrays.put(device.getDeviceId(), array);
    }

    public String getName() {
        return name == null ? "" : name;
    }

    public ParameterType getType() {
        return type;
    }

    public NDArray getArray() {
        return getArray(devices[0].getDeviceId());
    }

    public NDArray getArray(Device device) {
        return getArray(device.getDeviceId());
    }

    public NDArray getArray(int deviceId) {
        if (!isInitialized()) {
            throw new IllegalStateException("The array has not been initialized");
        }
        return arrays.get(deviceId);
    }

    public boolean isInitialized() {
        return arrays != null && !arrays.isEmpty() && arrays.size() == devices.length;
    }

    public void setInitializer(
            NDManager manager, Initializer initializer, boolean overwrite, Device[] devices) {
        this.manager = manager;
        if (overwrite || this.initializer == null) {
            this.initializer = initializer;
        }
        this.devices = devices;
    }

    public void reinitialize(Device[] devices) {
        if (!isInitialized()) {
            throw new IllegalStateException("This parameter is not initialized");
        }
        Objects.requireNonNull(initializer, "No initializer has been set");
        this.devices = devices;
        NDArray oldArray = arrays.get(0);
        NDArray newArray =
                initializer.initialize(
                        manager, oldArray.getShape(), oldArray.getDataType(), devices[0]);
        oldArray.close();
        arrays.put(devices[0].getDeviceId(), newArray);
        // multi-gpu
        if (devices.length > 1) {
            for (int i = 1; i < devices.length; i++) {
                NDArray arrayCopy = newArray.asInDevice(devices[i], true);
                arrayCopy.attachGradient();
                // close the old array
                arrays.get(devices[i].getDeviceId()).close();
                arrays.put(devices[i].getDeviceId(), arrayCopy);
            }
        }
    }

    public void reinitialize() {
        reinitialize(devices);
    }

    public void initialize(NDList inputs, boolean overwrite) {
        Objects.requireNonNull(initializer, "No initializer has been set");

        if (isInitialized()) {
            if (!overwrite) {
                throw new IllegalStateException("This parameter is already initialized");
            }
            // TODO: close old array
        }
        Shape[] shapes = new Shape[inputs.size()];
        for (int i = 0; i < shapes.length; ++i) {
            shapes[i] = inputs.get(i).getShape();
        }
        NDArray array =
                initializer.initialize(
                        manager,
                        block.getParameterShape(name, shapes),
                        inputs.head().getDataType(),
                        devices[0]);
        array.attachGradient();
        arrays.put(devices[0].getDeviceId(), array);
        // multi gpu
        if (devices.length > 1) {
            for (int i = 1; i < devices.length; i++) {
                NDArray arrayCopy = array.asInDevice(devices[i], true);
                arrayCopy.attachGradient();
                arrays.put(devices[i].getDeviceId(), arrayCopy);
            }
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

        dos.writeUTF(arrays.get(0).getSparseFormat().name());
        dos.writeUTF(arrays.get(0).getDataType().name());

        Shape shape = arrays.get(0).getShape();
        dos.write(shape.getEncoded());

        ByteBuffer bb = arrays.get(0).toByteBuffer();
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

        arrays.put(0, manager.create(dataType.asDataType(data), shape));
    }

    @Override
    public void close() {
        if (arrays != null && !arrays.isEmpty()) {
            for (NDArray array : arrays.values()) {
                array.close();
            }
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
                && Objects.equals(arrays.get(0), parameter.arrays.get(0));
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, type, arrays.get(0));
    }
}

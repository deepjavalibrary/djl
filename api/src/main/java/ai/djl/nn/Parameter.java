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
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.training.initializer.Initializer;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Objects;
import java.util.UUID;

/**
 * {@code Parameter} is a container class that holds a learnable parameter of a model.
 *
 * <p>Every {@code Parameter} is associated with a {@link Block}. The output of the block's forward
 * function depends on the values in the {@code Parameter}. During training, the values in the
 * {@code Parameter} are updated to reflect the training data. This process forms the crux of
 * learning.
 */
public class Parameter implements AutoCloseable {

    private static final byte VERSION = 1;

    private String id;
    private String name;
    private Block block;
    private ParameterType type;
    private DataType mandatoryDataType;
    private Initializer initializer;
    private NDArray array;
    private boolean requiresGrad;
    private SparseFormat gradientFormat;

    /**
     * Creates a {@code Parameter} with the given name, and parameter type, and associated with the
     * given {@link Block}.
     *
     * @param name the name of the {@code Parameter}
     * @param block the block with which this {@code Parameter} is associated
     * @param type the type of this {@code Parameter}
     */
    public Parameter(String name, Block block, ParameterType type) {
        this(name, block, type, true, SparseFormat.DENSE);
    }

    /**
     * Creates a {@code Parameter} with the given name, and parameter type, and associated with the
     * given {@link Block}.
     *
     * @param name the name of the {@code Parameter}
     * @param block the block with which this {@code Parameter} is associated
     * @param type the type of this {@code Parameter}
     * @param requiresGrad whether this {@code Parameter} needs to compute gradients
     */
    public Parameter(String name, Block block, ParameterType type, boolean requiresGrad) {
        this(name, block, type, requiresGrad, SparseFormat.DENSE);
    }

    /**
     * Creates a {@code Parameter} with the given name, and parameter type, and associated with the
     * given {@link Block}.
     *
     * @param name the name of the {@code Parameter}
     * @param block the block with which this {@code Parameter} is associated
     * @param type the type of this {@code Parameter}
     * @param requireGrad whether this {@code Parameter} needs to compute gradients
     * @param gradientFormat the {@link SparseFormat} of the gradient array
     */
    public Parameter(
            String name,
            Block block,
            ParameterType type,
            boolean requireGrad,
            SparseFormat gradientFormat) {
        this.id = UUID.randomUUID().toString();
        this.name = name;
        this.block = block;
        this.type = type;
        this.requiresGrad = requireGrad;
        this.initializer = type.getInitializer();
        this.gradientFormat = gradientFormat;
    }

    /**
     * Gets the ID of this {@code Parameter}.
     *
     * @return the ID of this {@code Parameter}
     */
    public String getId() {
        return id;
    }

    /**
     * Gets the name of this {@code Parameter}.
     *
     * @return the name of this {@code Parameter}
     */
    public String getName() {
        return name == null ? "" : name;
    }

    /**
     * Gets the type of this {@code Parameter}.
     *
     * @return the type of this {@code Parameter}
     */
    public ParameterType getType() {
        return type;
    }

    /**
     * Sets the values of this {@code Parameter}.
     *
     * @param array the {@link NDArray} that contains values of this {@code Parameter}
     */
    public void setArray(NDArray array) {
        this.array = array;
        array.setName(name);
    }

    /**
     * Gets the values of this {@code Parameter} as an {@link NDArray}.
     *
     * @return an {@link NDArray} that contains values of this {@code Parameter}
     */
    public NDArray getArray() {
        if (!isInitialized()) {
            throw new IllegalStateException("The array has not been initialized");
        }
        return array;
    }

    /**
     * Returns whether this parameter needs gradients to be computed.
     *
     * @return whether this parameter needs gradients to be computed
     */
    public boolean requireGradient() {
        return requiresGrad;
    }

    /**
     * Sets the mandatory data type for this {@code Parameter}.
     *
     * @param mandatoryDataType the mandatory data type for this {@code Parameter}
     */
    public void setMandatoryDataType(DataType mandatoryDataType) {
        this.mandatoryDataType = mandatoryDataType;
    }

    /**
     * Checks if this {@code Parameter} is initialized.
     *
     * @return {@code true} if this {@code Parameter} is initialized
     */
    public boolean isInitialized() {
        return array != null;
    }

    /**
     * Sets the {@link Initializer} for this {@code Parameter}, if not already set. If overwrite
     * flag is true, sets the initializer regardless.
     *
     * @param initializer the initializer to be set
     * @param overwrite if true, set the initializer regardless of whether its already set or not
     */
    public void setInitializer(Initializer initializer, boolean overwrite) {
        if (overwrite || this.initializer == null) {
            this.initializer = initializer;
        }
    }

    /**
     * Initializes the parameter with the given {@link NDManager}, with given {@link DataType} for
     * the given expected input shapes.
     *
     * @param manager an NDManager to create the arrays
     * @param dataType the datatype of the {@code Parameter}
     * @param inputShapes the expected input shapes
     */
    public void initialize(NDManager manager, DataType dataType, Shape[] inputShapes) {
        Objects.requireNonNull(initializer, "No initializer has been set");
        if (!isInitialized()) {
            Shape shape = block.getParameterShape(name, inputShapes);
            array =
                    initializer.initialize(
                            manager,
                            shape,
                            mandatoryDataType == null ? dataType : mandatoryDataType);
            array.setName(name);
        }

        if (requireGradient()) {
            array.attachGradient(gradientFormat);
        }
    }

    /**
     * Writes the parameter NDArrays to the given output stream.
     *
     * @param dos the output stream to write to
     * @throws IOException if the write operation fails
     */
    public void save(DataOutputStream dos) throws IOException {
        if (!isInitialized()) {
            dos.writeChar('N');
            return;
        }

        dos.writeChar('P');
        dos.writeByte(VERSION);
        dos.writeUTF(getName());
        dos.write(array.encode());
    }

    /**
     * Loads parameter NDArrays from InputStream.
     *
     * <p>Currently, we cannot deserialize into the exact subclass of NDArray. The SparseNDArray
     * will be loaded as NDArray only.
     *
     * @param manager the NDManager
     * @param dis the InputStream
     * @throws IOException if failed to read
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

        array = manager.decode(dis);
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

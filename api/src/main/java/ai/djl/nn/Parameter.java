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
import ai.djl.training.initializer.XavierInitializer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Objects;

/**
 * {@code Parameter} is a container class that holds a learnable parameter of a model.
 *
 * <p>Every {@code Parameter} is associated with a {@link Block}. The output of the block's forward
 * function depends on the values in the {@code Parameter}. During training, the values in the
 * {@code Parameter} are updated to reflect the training data. This process forms the crux of
 * learning.
 *
 * @see <a href="https://d2l.djl.ai/chapter_deep-learning-computation/parameters.html">The D2L
 *     chapter on parameter management</a>
 */
public class Parameter implements AutoCloseable {

    private static final byte VERSION = 1;

    private String id;
    private String name;
    private Shape shape;
    private Type type;
    private Initializer initializer;
    private NDArray array;
    private boolean requiresGrad;

    Parameter(Builder builder) {
        this.id = NDManager.nextUid();
        this.name = builder.name;
        this.shape = builder.shape;
        this.type = builder.type;
        this.array = builder.array;
        this.requiresGrad = builder.requiresGrad;
        this.initializer =
                (builder.initializer != null) ? builder.initializer : type.getInitializer();
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
    public Type getType() {
        return type;
    }

    /**
     * Sets the values of this {@code Parameter}.
     *
     * @param array the {@link NDArray} that contains values of this {@code Parameter}
     */
    public void setArray(NDArray array) {
        if (shape != null) {
            throw new IllegalStateException("array has been set! Use either setArray or setShape");
        }
        this.array = array;
        shape = array.getShape();
        array.setName(name);
    }

    /**
     * Sets the shape of this {@code Parameter}.
     *
     * @param shape the shape of this {@code Parameter}
     */
    public void setShape(Shape shape) {
        if (array != null) {
            throw new IllegalStateException("array has been set! Use either setArray or setShape");
        }
        this.shape = shape;
    }

    /**
     * Gets the shape of this {@code Parameter}.
     *
     * @return the shape of this {@code Parameter}
     */
    public Shape getShape() {
        return shape;
    }

    /**
     * Gets the values of this {@code Parameter} as an {@link NDArray}.
     *
     * @return an {@link NDArray} that contains values of this {@code Parameter}
     */
    public NDArray getArray() {
        if (!isInitialized()) {
            throw new UninitializedParameterException(
                    "The array for parameter \"" + getName() + "\" has not been initialized");
        }
        return array;
    }

    /**
     * Returns whether this parameter needs gradients to be computed.
     *
     * @return whether this parameter needs gradients to be computed
     */
    public boolean requiresGradient() {
        return requiresGrad;
    }

    /**
     * Freezes or unfreezes the parameter for training.
     *
     * <p>Sometimes during training, especially during transfer learning, it is typical to train
     * only part of the model. For this, the freeze can be used to prevent certain parts from being
     * trained.
     *
     * <p>This modifies the {@link #requiresGradient()} of the parameter.
     *
     * @param freeze true if the parameter should be frozen ({@code freeze == !requiresGradient()})
     */
    public void freeze(boolean freeze) {
        requiresGrad = !freeze;
        if (array != null) {
            // array can be null if block is loaded and then cleared
            array.setRequiresGradient(requiresGrad);
        }
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
     */
    public void setInitializer(Initializer initializer) {
        this.initializer = initializer;
    }

    /**
     * Returns the {@link Initializer} for this {@code Parameter}, if not already set. If overwrite
     * flag is true, sets the initializer regardless.
     *
     * @return the initializer of this {@code Parameter}
     */
    public Initializer getInitializer() {
        return initializer;
    }

    /**
     * Initializes the parameter with the given {@link NDManager}, with given {@link DataType} for
     * the given expected input shapes.
     *
     * @param manager an NDManager to create the arrays
     * @param dataType the datatype of the {@code Parameter}
     */
    public void initialize(NDManager manager, DataType dataType) {
        // Param is attached to an array not null
        if (!isInitialized()) {
            // Params in a PtSymbolBlock is set during model loading and its isInitialized()=true.
            // Shouldn't further initialize it.
            Objects.requireNonNull(initializer, "No initializer has been set");
            // Params in a PtSymbolBlock can have null shape, but are still initialized (i.e. param
            // is attached to an array not null)
            Objects.requireNonNull(shape, "No parameter shape has been set");
            array = initializer.initialize(manager, shape, dataType);
            array.setName(name);
        }

        if (requiresGradient()) {
            array.setRequiresGradient(true);
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
        // set the shape of the parameter and prepare() can be skipped
        shape = array.getShape();
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (array != null) {
            array.close();
            array = null;
        }
    }

    /**
     * Creates a builder to build a {@code Parameter}.
     *
     * <p>The methods start with {@code set} are required fields, and {@code opt} for optional
     * fields.
     *
     * @return a new builder
     */
    public static Parameter.Builder builder() {
        return new Parameter.Builder();
    }

    /** Enumerates the types of {@link Parameter}. */
    public enum Type {
        WEIGHT(
                new XavierInitializer(
                        XavierInitializer.RandomType.GAUSSIAN, XavierInitializer.FactorType.IN, 2)),
        BIAS(Initializer.ZEROS),
        GAMMA(Initializer.ONES),
        BETA(Initializer.ZEROS),
        RUNNING_MEAN(Initializer.ZEROS),
        RUNNING_VAR(Initializer.ONES),
        OTHER(null);

        private final transient Initializer initializer;

        Type(Initializer initializer) {
            this.initializer = initializer;
        }

        /**
         * Gets the {@link Initializer} of this {@code ParameterType}.
         *
         * @return the {@link Initializer} of this {@code ParameterType}
         */
        public Initializer getInitializer() {
            return initializer;
        }
    }

    /** A Builder to construct a {@code Parameter}. */
    public static final class Builder {
        String name;
        Shape shape;
        Type type;
        Initializer initializer;
        NDArray array;
        boolean requiresGrad = true;

        /**
         * Sets the name of the {@code Parameter}.
         *
         * @param name the name of the {@code Parameter}
         * @return this {@code Parameter}
         */
        public Builder setName(String name) {
            this.name = name;
            return this;
        }

        /**
         * Sets the {@code Type} of the {@code Parameter}.
         *
         * @param type the {@code Type} of the {@code Parameter}
         * @return this {@code Parameter}
         */
        public Builder setType(Type type) {
            this.type = type;
            return this;
        }

        /**
         * Sets the shape of the {@code Parameter}.
         *
         * @param shape the shape of the {@code Parameter}
         * @return this {@code Parameter}
         */
        public Builder optShape(Shape shape) {
            this.shape = shape;
            return this;
        }

        /**
         * Sets the Initializer of the {@code Parameter}.
         *
         * @param initializer the Initializer of the {@code Parameter}
         * @return this {@code Parameter}
         */
        public Builder optInitializer(Initializer initializer) {
            this.initializer = initializer;
            return this;
        }

        /**
         * Sets the array of the {@code Parameter}.
         *
         * @param array the array of the {@code Parameter}
         * @return this {@code Parameter}
         */
        public Builder optArray(NDArray array) {
            this.array = array;
            return this;
        }

        /**
         * Sets if the {@code Parameter} requires gradient.
         *
         * @param requiresGrad if the {@code Parameter} requires gradient
         * @return this {@code Parameter}
         */
        public Builder optRequiresGrad(boolean requiresGrad) {
            this.requiresGrad = requiresGrad;
            return this;
        }

        /**
         * Builds a {@code Parameter} instance.
         *
         * @return the {@code Parameter} instance
         */
        public Parameter build() {
            return new Parameter(this);
        }
    }
}

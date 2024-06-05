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
package ai.djl.translate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * A class that adapts Java's ByteBuffer to store data of various types in a structured manner. It
 * supports conversion and storage of different data types into a ByteBuffer.
 *
 * <p><b>Background:</b><br>
 * Typically, converting input data into the NDArray format accepted by models is a tedious and
 * time-consuming process. Generally, the input data for a Translator is copied into a temporary
 * input, then the NDArray is constructed using the {@link
 * ai.djl.ndarray.NDManager#create(java.nio.Buffer, ai.djl.ndarray.types.Shape,
 * ai.djl.ndarray.types.DataType)} method. DJL will copy the data from an array into non-direct
 * memory, and then from non-direct memory into direct memory. Finally, this direct memory is used
 * to construct the underlying tensor (this is a general statement; in TensorFlow, the direct memory
 * is actually copied again). This process involves a significant amount of unnecessary overhead.
 *
 * <p><b>Optimization:</b><br>
 * If you're looking to optimize the time cost of this process, you might consider using
 * ByteBufferAdapter. However, I must caution you that using this class requires a certain
 * understanding of the DJL framework, otherwise it might lead to issues with native memory.
 *
 * <p><b>Efficient Translator Implementation:</b><br>
 * An efficient translator implementation that demonstrates how to assemble batches manually to
 * leverage the performance benefits of direct memory. This example shows the usage of
 * ByteBufferAdapter for handling input data in a performant manner.
 *
 * <p>Note: By implementing the batch assembly logic within the translator and returning null from
 * getBatchifier(), we can effectively utilize direct memory for performance gains.
 *
 * <p><b>Example Usage:</b>
 *
 * <pre>{@code
 * import ai.djl.ndarray.types.DataType;
 * import ai.djl.translate.Batchifier;
 * import ai.djl.translate.Translator;
 * import ai.djl.translate.TranslatorContext;
 * import java.util.List;
 *
 * public class EfficientTranslator implements Translator<List<float[]>, List<float[]>> {
 *     @Override
 *     public Batchifier getBatchifier() {
 *         // Attention here: returning null to indicate manual batch handling within the translator
 *         return null;
 *     }
 *
 *     @Override
 *     public List<float[]> processOutput(TranslatorContext ctx, NDList list) throws Exception {
 *         // Implement output processing logic here
 *         return null;
 *     }
 *
 *     @Override
 *     public NDList processInput(TranslatorContext ctx, List<float[]> inputs) throws Exception {
 *         // Example of using ByteBufferAdapter for efficient input data handling
 *         final ByteBufferAdapter byteBufferAdapter = ByteBufferAdapter
 *                 .builder()
 *                 .setDataType(DataType.FLOAT32)
 *                 .setShape(10, 32) // Example shape, adjust according to actual needs
 *                 .setDirect() // Opting for direct memory for performance
 *                 .build();
 *
 *         // Populating the ByteBufferAdapter with input data
 *         for (float[] nums : inputs) {
 *             for (float num : nums) {
 *                 byteBufferAdapter.putObject(num);
 *             }
 *         }
 *
 *         // Converting the populated ByteBufferAdapter to NDArray for further processing
 *         final NDArray ndArray = byteBufferAdapter.toNDArray(ctx.getNDManager());
 *         return new NDList(ndArray);
 *     }
 * }
 * }</pre>
 */
public class ByteBufferAdapter implements AutoCloseable {
    /**
     * Typically, you should set `java.nio.ByteBuffer` to direct memory to avoid additional copying.
     * However, it's important to note that after using the {@link ByteBufferAdapter#toNDArray}
     * method, the management of direct memory is handed over to {@link ai.djl.ndarray.NDManager};
     * you might consider setting up a cleaner for active release. But, pay special attention that
     * most engines will use this direct memory as the underlying tensor's data area directly,
     * hence, if you misuse the cleaner for release, it could lead to severe consequences. For
     * instance, engines like PyTorch, ONNX Runtime, and LightGBM directly use this area as the
     * tensor's memory region, meaning you cannot immediately use a cleaner to release the native
     * memory after {@link ByteBufferAdapter#toNDArray} has completed, unless the associated NDArray
     * is no longer needed; however, TensorFlow will perform an additional copy, so you can use a
     * cleaner to release the native memory right after {@link ByteBufferAdapter#toNDArray}
     * execution is completed. Therefore, generally speaking, unless you are very certain that the
     * release by the cleaner is absolutely necessary, and you have a deep understanding of the
     * relevant code, it is not recommended using a cleaner.
     */
    private static final Logger logger = LoggerFactory.getLogger(ByteBufferAdapter.class);

    private final Shape shape;
    private final DataType dataType;
    private ByteBuffer byteBuffer;
    private final ConvertorHolder.Convertor convertor;
    private final Runnable cleaner;

    private ByteBufferAdapter(Builder builder) {
        this.shape = builder.shape;
        this.dataType = builder.dataType;
        int size = (int) (this.shape.size() * this.dataType.getNumOfBytes());
        if (builder.isDirect) {
            this.byteBuffer = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder());
        } else {
            this.byteBuffer = ByteBuffer.allocate(size).order(ByteOrder.nativeOrder());
        }
        this.convertor = ConvertorHolder.getConvertor(this.dataType);
        this.cleaner = builder.cleaner;
    }

    /**
     * Creates a new instance of {@link Builder}.
     *
     * <p>This method offers a convenient way to construct a {@link ByteBufferAdapter} object. Users
     * can customize the desired configuration by chaining calls to methods provided by {@link
     * Builder}, ultimately building a {@link ByteBufferAdapter} instance.
     *
     * @return Returns a new instance of {@link Builder} for constructing a {@link
     *     ByteBufferAdapter}.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Puts an object into the ByteBuffer after converting it to the appropriate type.
     *
     * @param value The object to be stored.
     */
    public void putObject(Object value) {
        value = convertor.convert(value);
        unsafePutObject(value);
    }

    /**
     * Puts an object into the ByteBuffer after converting it to the appropriate type.
     *
     * <p>This method directly stores the object into the ByteBuffer using the converter to match
     * the expected data type. It bypasses null or type compatibility checks, so the caller should
     * ensure the value's validity and compatibility beforehand.
     *
     * @param value The object to be stored, which should be non-null and compatible with the target
     *     data type before calling.
     */
    public void unsafePutObject(Object value) {
        convertor.put(byteBuffer, value);
    }

    /**
     * Converts the data in the ByteBuffer to an NDArray.
     *
     * @param manager The NDManager to create the NDArray.
     * @return The created NDArray.
     */
    public NDArray toNDArray(NDManager manager) {
        if (manager == null) {
            throw new IllegalArgumentException("NDManager cannot be null.");
        }
        if (byteBuffer == null) {
            throw new IllegalArgumentException("ByteBuffer cannot be null.");
        }
        if (shape == null) {
            throw new IllegalArgumentException("Shape cannot be null.");
        }
        if (dataType == null) {
            throw new IllegalArgumentException("DataType cannot be null.");
        }
        byteBuffer.rewind();
        return manager.create(byteBuffer, shape, dataType);
    }

    /**
     * For non-direct memory, a cleaner is not necessary. For direct memory, it's generally not
     * required either. However, if you believe that the Garbage Collector (GC) is not reclaiming
     * native memory in a timely manner, you can implement the logic for releasing native memory
     * here.
     *
     * <p>Future modifications should be cautious not to hardcode the logic for releasing native
     * memory in the source code, as this may cause compatibility issues with different JDK
     * versions. It's recommended that users implement this logic themselves.
     */
    @Override
    public void close() {
        if (cleaner != null) {
            try {
                cleaner.run();
            } catch (Exception e) {
                logger.error("An error occurred while running the cleaner.", e);
            }
        }
        byteBuffer = null;
    }

    /** Builder class for ByteBufferAdapter. */
    public static class Builder {
        private boolean isDirect = true;
        private Shape shape;
        private DataType dataType;
        private Runnable cleaner;

        /**
         * Sets the ByteBuffer to use direct memory.
         *
         * <p>This method configures the builder to allocate the ByteBuffer using direct memory,
         * which can enhance performance. Direct memory can reduce the copying between Java heap and
         * native memory, potentially leading to more efficient I/O operations.
         *
         * @return The builder instance for chaining.
         */
        public Builder setDirect() {
            this.isDirect = true;
            return this;
        }

        /**
         * Sets the ByteBuffer to use non-direct memory.
         *
         * <p>This method configures the builder to allocate the ByteBuffer using non-direct memory,
         * which may be useful in scenarios where direct memory does not offer significant
         * performance benefits or where direct memory usage is not desired. Non-direct memory is
         * managed by the JVM and can be easier to deal with in terms of memory release and garbage
         * collection.
         *
         * @return The builder instance for chaining.
         */
        public Builder setNoDirect() {
            this.isDirect = false;
            return this;
        }

        /**
         * Sets the shape for the ByteBufferAdapter.
         *
         * <p>This method configures the shape of the data to be stored in the ByteBufferAdapter.
         * The shape is essential for interpreting the data correctly during further processing. It
         * ensures that the data dimensions are accurately represented.
         *
         * @param shape The dimensions of the shape as a variable number of long arguments. Each
         *     argument represents the size of the dimension.
         * @return The builder instance for chaining method calls.
         * @throws IllegalArgumentException if the shape is null or has no dimensions, ensuring that
         *     a valid shape is always provided.
         */
        public Builder setShape(long... shape) {
            if (shape == null || shape.length == 0) {
                throw new IllegalArgumentException("Shape cannot be null or empty");
            }
            this.shape = new Shape(shape);
            return this;
        }

        /**
         * Sets the shape for the ByteBufferAdapter.
         *
         * <p>This method configures the shape of the data to be stored in the ByteBufferAdapter.
         * The shape is crucial for correctly interpreting the data during further processing,
         * ensuring that the data dimensions are accurately represented.
         *
         * @param shape The shape of the data as a Shape object.
         * @return The builder instance for chaining method calls.
         */
        public Builder setShape(Shape shape) {
            this.shape = shape;
            return this;
        }

        /**
         * Sets the data type for the ByteBufferAdapter.
         *
         * <p>This method specifies the type of data that will be stored in the ByteBufferAdapter,
         * which is essential for ensuring that the data is correctly processed and interpreted.
         *
         * @param dataType The data type of the stored data.
         * @return The builder instance for chaining method calls.
         */
        public Builder setDataType(DataType dataType) {
            this.dataType = dataType;
            return this;
        }

        /**
         * Sets a cleaner for the ByteBufferAdapter.
         *
         * <p>This method allows setting a custom cleaner runnable that can be used to release
         * resources or perform cleanup when the ByteBufferAdapter is closed. This is particularly
         * useful for direct memory buffers where manual cleanup might be necessary.
         *
         * @param cleaner A runnable that defines the cleanup logic.
         * @return The builder instance for chaining method calls.
         */
        public Builder optCleaner(Runnable cleaner) {
            this.cleaner = cleaner;
            return this;
        }

        /**
         * Builds and returns a new ByteBufferAdapter instance.
         *
         * <p>This method finalizes the configuration of the ByteBufferAdapter by first performing
         * validation checks through the {@code check} method. If the configuration passes all
         * checks, it then creates a new instance of ByteBufferAdapter using the current state of
         * the Builder.
         *
         * @return A new instance of ByteBufferAdapter configured according to the Builder settings.
         */
        public ByteBufferAdapter build() {
            check();
            return new ByteBufferAdapter(this);
        }

        private void check() {
            if (this.dataType == null) {
                throw new IllegalArgumentException("Data type cannot be null");
            }

            if (!ConvertorHolder.support(dataType)) {
                Set<DataType> supportedDataTypes = ConvertorHolder.getSupportedDataTypes();
                throw new IllegalArgumentException(
                        "Unsupported data type: "
                                + dataType
                                + ". Supported data types are: "
                                + supportedDataTypes);
            }

            if (this.shape == null) {
                throw new IllegalArgumentException("Shape cannot be null");
            }

            long totalSize = 1;
            for (long dim : this.shape.getShape()) {
                if (dim <= 0 || dim > Integer.MAX_VALUE) {
                    throw new IllegalArgumentException(
                            "Each dimension of the shape must be a positive number and not exceed"
                                    + " the maximum array size in Java");
                }

                if (totalSize > (Integer.MAX_VALUE / dim)) {
                    throw new IllegalArgumentException(
                            "The total size of the shape exceeds the maximum array size in Java");
                }
                totalSize *= dim;
            }
        }
    }

    /** Holds converters for different data types. */
    static class ConvertorHolder {
        private static final Map<DataType, Convertor> CONVERTOR_MAP = new HashMap<>();

        static {
            CONVERTOR_MAP.put(DataType.FLOAT32, new FloatConvert());
            CONVERTOR_MAP.put(DataType.FLOAT64, new DoubleConvert());
            CONVERTOR_MAP.put(DataType.INT32, new IntConvert());
            CONVERTOR_MAP.put(DataType.INT8, new ByteConvert());
            CONVERTOR_MAP.put(DataType.INT64, new LongConvert());
            CONVERTOR_MAP.put(DataType.BOOLEAN, new BooleanConvert());
        }

        public static Convertor getConvertor(DataType dataType) {
            return CONVERTOR_MAP.get(dataType);
        }

        public static boolean support(DataType dataType) {
            return CONVERTOR_MAP.containsKey(dataType);
        }

        public static Set<DataType> getSupportedDataTypes() {
            return CONVERTOR_MAP.keySet();
        }

        interface Convertor {
            Object convert(Object value);

            void put(ByteBuffer byteBuffer, Object value);
        }

        static class IntConvert implements Convertor {
            @Override
            public Object convert(Object value) {
                if (value == null) {
                    return null;
                } else if (value instanceof Integer) {
                    return (Integer) value;
                } else if (value instanceof Number) {
                    return ((Number) value).intValue();
                } else {
                    throw new IllegalArgumentException(
                            "Unsupported data type for conversion: "
                                    + value.getClass().getName()
                                    + ". Expected Integer value.");
                }
            }

            @Override
            public void put(ByteBuffer byteBuffer, Object value) {
                if (value == null) {
                    throw new NullPointerException(
                            "Cannot put a null value into a ByteBuffer as a int.");
                }
                if (!(value instanceof Integer)) {
                    throw new IllegalArgumentException(
                            "Expected an Integer value for putting into ByteBuffer, but received: "
                                    + value.getClass().getName());
                }
                byteBuffer.putInt((Integer) value);
            }
        }

        static class FloatConvert implements Convertor {
            @Override
            public Object convert(Object value) {
                if (value == null) {
                    return null;
                } else if (value instanceof Float) {
                    return (Float) value;
                } else if (value instanceof Number) {
                    return ((Number) value).floatValue();
                } else {
                    throw new IllegalArgumentException(
                            "Unsupported data type for conversion: "
                                    + value.getClass().getName()
                                    + ". Expected Float value.");
                }
            }

            @Override
            public void put(ByteBuffer byteBuffer, Object value) {
                if (value == null) {
                    throw new NullPointerException(
                            "Cannot put a null value into a ByteBuffer as a float.");
                }
                if (!(value instanceof Float)) {
                    throw new IllegalArgumentException(
                            "Expected a Float value for putting into ByteBuffer, but received: "
                                    + value.getClass().getName());
                }
                byteBuffer.putFloat((Float) value);
            }
        }

        static class DoubleConvert implements Convertor {
            @Override
            public Object convert(Object value) {
                if (value == null) {
                    return null;
                } else if (value instanceof Double) {
                    return (Double) value;
                } else if (value instanceof Number) {
                    return ((Number) value).doubleValue();
                } else {
                    throw new IllegalArgumentException(
                            "Unsupported data type for conversion: "
                                    + value.getClass().getName()
                                    + ". Expected Double value.");
                }
            }

            @Override
            public void put(ByteBuffer byteBuffer, Object value) {
                if (value == null) {
                    throw new NullPointerException(
                            "Cannot put a null value into a ByteBuffer as a double.");
                }
                if (!(value instanceof Double)) {
                    throw new IllegalArgumentException(
                            "Expected a Double value for putting into ByteBuffer, but received: "
                                    + value.getClass().getName());
                }
                byteBuffer.putDouble((Double) value);
            }
        }

        static class ByteConvert implements Convertor {
            @Override
            public Object convert(Object value) {
                if (value == null) {
                    return null;
                } else if (value instanceof Byte) {
                    return (Byte) value;
                } else if (value instanceof Number) {
                    return ((Number) value).byteValue();
                } else {
                    throw new IllegalArgumentException(
                            "Unsupported data type for conversion: "
                                    + value.getClass().getName()
                                    + ". Expected Byte value.");
                }
            }

            @Override
            public void put(ByteBuffer byteBuffer, Object value) {
                if (value == null) {
                    throw new NullPointerException(
                            "Cannot put a null value into a ByteBuffer as a byte.");
                }
                if (!(value instanceof Byte)) {
                    throw new IllegalArgumentException(
                            "Expected a Byte value for putting into ByteBuffer, but received: "
                                    + value.getClass().getName());
                }
                byteBuffer.put((Byte) value);
            }
        }

        static class LongConvert implements Convertor {
            @Override
            public Object convert(Object value) {
                if (value == null) {
                    return null;
                } else if (value instanceof Long) {
                    return (Long) value;
                } else if (value instanceof Number) {
                    return ((Number) value).longValue();
                } else {
                    throw new IllegalArgumentException(
                            "Unsupported data type for conversion: "
                                    + value.getClass().getName()
                                    + ". Expected Long value.");
                }
            }

            @Override
            public void put(ByteBuffer byteBuffer, Object value) {
                if (value == null) {
                    throw new NullPointerException(
                            "Cannot put a null value into a ByteBuffer as a long.");
                }
                if (!(value instanceof Long)) {
                    throw new IllegalArgumentException(
                            "Expected a Long value for putting into ByteBuffer, but received: "
                                    + value.getClass().getName());
                }
                byteBuffer.putLong((Long) value);
            }
        }

        static class BooleanConvert implements Convertor {
            @Override
            public Object convert(Object value) {
                if (value == null) {
                    return null;
                } else if (value instanceof Boolean) {
                    return value;
                } else if (value instanceof Number) {
                    return ((Number) value).intValue() != 0;
                } else if (value instanceof String) {
                    return Boolean.parseBoolean((String) value);
                } else {
                    throw new IllegalArgumentException(
                            "Unsupported data type for conversion: "
                                    + value.getClass().getName()
                                    + ". Expected Boolean value.");
                }
            }

            @Override
            public void put(ByteBuffer byteBuffer, Object value) {
                if (value == null) {
                    throw new NullPointerException(
                            "Cannot put a null value into a ByteBuffer as a bool.");
                }
                if (!(value instanceof Boolean)) {
                    throw new IllegalArgumentException(
                            "Expected a Boolean value for putting into ByteBuffer, but received: "
                                    + value.getClass().getName());
                }
                byteBuffer.put((byte) ((Boolean) value ? 1 : 0));
            }
        }
    }
}

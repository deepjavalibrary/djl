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
package ai.djl.ndarray;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.engine.EngineException;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.PairList;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

/**
 * NDArray managers are used to create <I>NDArrays</I> (n-dimensional array on native engine).
 *
 * <p>NDManager is implemented in each deep learning {@link Engine}. {@link NDArray}s are resources
 * that are allocated in each deep learning engine's native memory space. NDManager is the key class
 * that manages these native resources.
 *
 * <p>NDArray can only be created through NDManager. By default, NDArray's lifecycle is attached to
 * the creator NDManager. NDManager itself implements {@link AutoCloseable}. When NDManager is
 * closed, all the resource associated with it will be closed as well.
 *
 * <p>A typical place to obtain NDManager is in {@link Translator#processInput(TranslatorContext,
 * Object)} or {@link Translator#processOutput(TranslatorContext, NDList)}.
 *
 * <p>The following is an example of how to use NDManager:
 *
 * <pre>
 * public class MyTranslator implements Translator&lt;FloatBuffer, String&gt; {
 *
 *     &#064;Override
 *     public NDList processInput(TranslatorContext ctx, FloatBuffer input) {
 *         <b>NDManager manager = ctx.getNDManager();</b>
 *         NDArray array = <b>manager</b>.create(shape);
 *         array.set(input);
 *         return new NDList(array);
 *     } // NDArrays created in this method will be closed after method return.
 * }
 * </pre>
 *
 * <p>NDManager has a hierarchical structure; it has a single parent NDManager and has child
 * NDManagers. When the parent NDManager is closed, all children will be closed as well.
 *
 * <p>The DJL engine manages NDManager's lifecycle by default. You only need to manage the user
 * created child NDManager. The child NDManager becomes useful when you create a large number of
 * temporary NDArrays and want to free the resources earlier than the parent NDManager's lifecycle.
 *
 * <p>The following is an example of such a use case:
 *
 * <pre>
 * public class MyTranslator implements Translator&lt;List&lt;FloatBuffer&gt;&gt;, String&gt; {
 *
 *     &#064;Override
 *     public NDList processInput(TranslatorContext ctx, List&lt;FloatBuffer&gt; input) {
 *         NDManager manager = ctx.getNDManager();
 *         NDArray array = manager.create(shape, dataType);
 *         for (int i = 0; i &lt; input.size(); ++i) {
 *             try (<b>NDManager childManager = manager.newSubManager()</b>) {
 *                  NDArray tmp = <b>childManager</b>.create(itemShape);
 *                  tmp.put(input.get(i);
 *                  array.put(i, tmp);
 *             } // NDArray <i>tmp</i> will be closed here
 *         }
 *         return new NDList(array);
 *     }
 * }
 * </pre>
 *
 * <p>You can also close an individual NDArray. NDManager won't close an NDArray that's already been
 * closed. In certain use cases, you might want to return an NDArray outside of NDManager's scope.
 *
 * @see NDArray
 * @see Translator
 * @see TranslatorContext#getNDManager()
 * @see <a
 *     href="https://github.com/deepjavalibrary/djl/blob/master/docs/development/memory_management.md">NDArray
 *     Memory Management Guide</a>
 */
public interface NDManager extends AutoCloseable {

    /**
     * Creates a new top-level {@code NDManager}.
     *
     * <p>{@code NDManager} will inherit default {@link Device}.
     *
     * @return a new top-level {@code NDManager}
     */
    static NDManager newBaseManager() {
        return Engine.getInstance().newBaseManager();
    }

    /**
     * Creates a new top-level {@code NDManager} with specified {@link Device}.
     *
     * @param device the default {@link Device}
     * @return a new top-level {@code NDManager}
     */
    static NDManager newBaseManager(Device device) {
        return Engine.getInstance().newBaseManager(device);
    }

    /**
     * Creates a new top-level {@code NDManager} with specified {@link Device} and engine.
     *
     * @param device the default {@link Device}
     * @param engineName the name of the engine
     * @return a new top-level {@code NDManager}
     */
    static NDManager newBaseManager(Device device, String engineName) {
        return Engine.getEngine(engineName).newBaseManager(device);
    }

    /**
     * Creates a new manager based on the given resource.
     *
     * @param resource the resource to use
     * @return a new memory scrope containing the array
     */
    static NDManager subManagerOf(NDResource resource) {
        return resource.getManager().newSubManager();
    }

    /**
     * Returns the default context used in Engine.
     *
     * <p>The default type is defined by whether the deep learning engine is recognizing GPUs
     * available on your machine. If there is no GPU available, CPU will be used.
     *
     * @return a {@link Device}
     */
    Device defaultDevice();

    /**
     * Allocates a new engine specific direct byte buffer.
     *
     * @param capacity the new buffer's capacity, in bytes
     * @return the new byte buffer
     */
    ByteBuffer allocateDirect(int capacity);

    /**
     * Creates a new {@code NDArray} if the input {@link NDArray} is from external engine.
     *
     * @param array the input {@code NDArray}
     * @return a new {@code NDArray} if the input {@code NDArray} is from external engine
     */
    NDArray from(NDArray array);

    /**
     * Creates an uninitialized instance of {@link DataType#FLOAT32} {@link NDArray} with specified
     * {@link Shape}.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(Shape shape) {
        return create(shape, DataType.FLOAT32, getDevice());
    }

    /**
     * Creates and initializes a scalar {@link NDArray}.
     *
     * @param data the {@link Number} that needs to be set
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(Number data) {
        if (data instanceof Integer) {
            return create(data.intValue());
        } else if (data instanceof Float) {
            return create(data.floatValue());
        } else if (data instanceof Double) {
            return create(data.doubleValue());
        } else if (data instanceof Long) {
            return create(data.longValue());
        } else if (data instanceof Byte) {
            return create(data.byteValue());
        } else {
            throw new IllegalArgumentException("Short conversion not supported!");
        }
    }

    /**
     * Creates and initializes a scalar {@link NDArray}.
     *
     * @param data the float that needs to be set
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(float data) {
        return create(new float[] {data}, new Shape());
    }

    /**
     * Creates and initializes a scalar {@link NDArray}.
     *
     * @param data the float data that needs to be set
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(int data) {
        return create(new int[] {data}, new Shape());
    }

    /**
     * Creates and initializes a scalar {@link NDArray}.
     *
     * @param data the double data that needs to be set
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(double data) {
        return create(new double[] {data}, new Shape());
    }

    /**
     * Creates and initializes a scalar {@link NDArray}.
     *
     * @param data the long data that needs to be set
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(long data) {
        return create(new long[] {data}, new Shape());
    }

    /**
     * Creates and initializes a scalar {@link NDArray}.
     *
     * @param data the byte data that needs to be set
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(byte data) {
        return create(new byte[] {data}, new Shape());
    }

    /**
     * Creates and initializes a scalar {@link NDArray}.
     *
     * @param data the boolean data that needs to be set
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(boolean data) {
        return create(new boolean[] {data}, new Shape());
    }

    /**
     * Creates and initializes a scalar {@link NDArray}.
     *
     * @param data the String data that needs to be set
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(String data) {
        return create(new String[] {data}, StandardCharsets.UTF_8, new Shape());
    }

    /**
     * Creates and initializes 1D {@link NDArray}.
     *
     * @param data the String data that needs to be set
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(String[] data) {
        return create(data, StandardCharsets.UTF_8);
    }

    /**
     * Creates and initializes 1D {@link NDArray}.
     *
     * @param data the String data that needs to be set
     * @param charset the charset to decode the string
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(String[] data, Charset charset) {
        return create(data, charset, new Shape(data.length));
    }

    /**
     * Creates a String {@link NDArray} based on the provided shape.
     *
     * @param data the flattened String array
     * @param shape the shape of the String NDArray
     * @return a new instance of {@code NDArray}
     */
    default NDArray create(String[] data, Shape shape) {
        return create(data, StandardCharsets.UTF_8, shape);
    }

    /**
     * Creates a String {@link NDArray} based on the provided shape.
     *
     * @param data the flattened String array
     * @param charset the charset to decode the string
     * @param shape the shape of the String NDArray
     * @return a new instance of {@code NDArray}
     */
    NDArray create(String[] data, Charset charset, Shape shape);

    /**
     * Creates and initializes a 1D {@link NDArray}.
     *
     * @param data the float array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(float[] data) {
        return create(data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 1D {@link NDArray}.
     *
     * @param data the float array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(int[] data) {
        return create(data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 1D {@link NDArray}.
     *
     * @param data the float array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(double[] data) {
        return create(data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 1D {@link NDArray}.
     *
     * @param data the float array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(long[] data) {
        return create(data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 1D {@link NDArray}.
     *
     * @param data the float array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(byte[] data) {
        return create(data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 1D {@link NDArray}.
     *
     * @param data the bool array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(boolean[] data) {
        return create(data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 2D {@link NDArray}.
     *
     * @param data the float array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(float[][] data) {
        FloatBuffer buffer = FloatBuffer.allocate(data.length * data[0].length);
        for (float[] d : data) {
            buffer.put(d);
        }
        buffer.rewind();
        return create(buffer, new Shape(data.length, data[0].length));
    }

    /**
     * Creates and initializes a 2D {@link NDArray}.
     *
     * @param data the float array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(int[][] data) {
        IntBuffer buffer = IntBuffer.allocate(data.length * data[0].length);
        for (int[] d : data) {
            buffer.put(d);
        }
        buffer.rewind();
        return create(buffer, new Shape(data.length, data[0].length));
    }

    /**
     * Creates and initializes a 2D {@link NDArray}.
     *
     * @param data the float array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(double[][] data) {
        DoubleBuffer buffer = DoubleBuffer.allocate(data.length * data[0].length);
        for (double[] d : data) {
            buffer.put(d);
        }
        buffer.rewind();
        return create(buffer, new Shape(data.length, data[0].length));
    }

    /**
     * Creates and initializes a 2-D {@link NDArray}.
     *
     * @param data the float array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(long[][] data) {
        LongBuffer buffer = LongBuffer.allocate(data.length * data[0].length);
        for (long[] d : data) {
            buffer.put(d);
        }
        buffer.rewind();
        return create(buffer, new Shape(data.length, data[0].length));
    }

    /**
     * Creates and initializes a 2-D {@link NDArray}.
     *
     * @param data the float array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(byte[][] data) {
        ByteBuffer buffer = ByteBuffer.allocate(data.length * data[0].length);
        for (byte[] d : data) {
            buffer.put(d);
        }
        buffer.rewind();
        return create(buffer, new Shape(data.length, data[0].length));
    }

    /**
     * Creates and initializes a 2-D {@link NDArray}.
     *
     * @param data the boolean array that needs to be set
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(boolean[][] data) {
        ByteBuffer buffer = ByteBuffer.allocate(data.length * data[0].length);
        for (boolean[] d : data) {
            for (boolean b : d) {
                buffer.put((byte) (b ? 1 : 0));
            }
        }
        buffer.rewind();
        return create(buffer, new Shape(data.length, data[0].length), DataType.BOOLEAN);
    }

    /**
     * Creates and initializes a {@link NDArray} with specified {@link Shape}.
     *
     * <p>{@link DataType} of the NDArray will determined by type of Buffer.
     *
     * @param data the data to initialize the {@code NDArray}
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(Buffer data, Shape shape) {
        DataType dataType = DataType.fromBuffer(data);
        return create(data, shape, dataType);
    }

    /**
     * Creates an uninitialized instance of {@link NDArray} with specified {@link Shape}, and {@link
     * DataType}.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    NDArray create(Shape shape, DataType dataType);

    /**
     * Creates and initializes an instance of {@link NDArray} with specified {@link Shape} and
     * {@link DataType}.
     *
     * @param data the data to initialize the {@link NDArray}
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(Buffer data, Shape shape, DataType dataType) {
        NDArray array = create(shape, dataType);
        array.set(data);
        return array;
    }

    /**
     * Creates and initializes an instance of {@link NDArray} with specified {@link Shape} and float
     * array.
     *
     * @param data the float array that needs to be set
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(float[] data, Shape shape) {
        return create(FloatBuffer.wrap(data), shape);
    }

    /**
     * Creates and initializes an instance of {@link NDArray} with specified {@link Shape} and int
     * array.
     *
     * @param data the float array that needs to be set
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(int[] data, Shape shape) {
        return create(IntBuffer.wrap(data), shape);
    }

    /**
     * Creates and initializes an instance of {@link NDArray} with specified {@link Shape} and
     * double array.
     *
     * @param data the float array that needs to be set
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(double[] data, Shape shape) {
        return create(DoubleBuffer.wrap(data), shape);
    }

    /**
     * Creates and initializes an instance of {@link NDArray} with specified {@link Shape} and long
     * array.
     *
     * @param data the float array that needs to be set
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(long[] data, Shape shape) {
        return create(LongBuffer.wrap(data), shape);
    }

    /**
     * Creates and initializes an instance of {@link NDArray} with specified {@link Shape} and byte
     * array.
     *
     * @param data the float array that needs to be set
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(byte[] data, Shape shape) {
        return create(ByteBuffer.wrap(data), shape);
    }

    /**
     * Creates and initializes an instance of {@link NDArray} with specified {@link Shape} and
     * boolean array.
     *
     * @param data the boolean array that needs to be set
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(boolean[] data, Shape shape) {
        byte[] byteData = new byte[data.length];
        for (int i = 0; i < data.length; i++) {
            byteData[i] = (byte) (data[i] ? 1 : 0);
        }
        return create(ByteBuffer.wrap(byteData), shape, DataType.BOOLEAN);
    }

    /**
     * Creates an uninitialized instance of {@link NDArray} with specified {@link Shape}, {@link
     * DataType} and {@link Device}.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    default NDArray create(Shape shape, DataType dataType, Device device) {
        if (device == null || device.equals(getDevice())) {
            return create(shape, dataType);
        }
        return newSubManager(device).create(shape, dataType);
    }

    /**
     * Creates a Compressed Sparse Row Storage (CSR) Format Matrix.
     *
     * @param data the data to set for the CSR Matrix
     * @param indptr the indptr array is what will help identify the rows where the data appears
     * @param indices the indices array stores the column index for each non-zero element in data
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    default NDArray createCSR(
            float[] data, long[] indptr, long[] indices, Shape shape, Device device) {
        return createCSR(FloatBuffer.wrap(data), indptr, indices, shape, device);
    }

    /**
     * Creates a Compressed Sparse Row Storage (CSR) Format Matrix.
     *
     * @param data the data to set for the CSR Matrix
     * @param indptr the indptr array is what will help identify the rows where the data appears
     * @param indices the indices array stores the column index for each non-zero element in data
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    default NDArray createCSR(
            Buffer data, long[] indptr, long[] indices, Shape shape, Device device) {
        if (device == null || device.equals(getDevice())) {
            return createCSR(data, indptr, indices, shape);
        }
        return newSubManager(device).createCSR(data, indptr, indices, shape);
    }

    /**
     * Creates a Compressed Sparse Row Storage (CSR) Format Matrix.
     *
     * @param data the data to set for the CSR Matrix
     * @param indptr the indptr array is what will help identify the rows where the data appears
     * @param indices the indices array stores the column index for each non-zero element in data
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    NDArray createCSR(Buffer data, long[] indptr, long[] indices, Shape shape);

    /**
     * Stores the matrix in row sparse format.
     *
     * @param data the data to set for the Row Sparse {@link NDArray}
     * @param dataShape the {@link Shape} of the data {@link NDArray}
     * @param indices the indices to store the data
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    default NDArray createRowSparse(
            Buffer data, Shape dataShape, long[] indices, Shape shape, Device device) {
        if (device == null || device.equals(getDevice())) {
            return createRowSparse(data, dataShape, indices, shape);
        }
        return newSubManager(device).createRowSparse(data, dataShape, indices, shape);
    }

    /**
     * Stores the matrix in row sparse format.
     *
     * @param data the data to set for the Row Sparse {@link NDArray}
     * @param dataShape the {@link Shape} of the data {@link NDArray}
     * @param indices the indices to store the data
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    NDArray createRowSparse(Buffer data, Shape dataShape, long[] indices, Shape shape);

    /**
     * Creates a Coordinate Format (COO) Matrix.
     *
     * @param data the data to set for the Coordinate format {@link NDArray}
     * @param indices the matrix represent indices
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    NDArray createCoo(Buffer data, long[][] indices, Shape shape);

    /**
     * Decodes {@link NDArray} through byte array.
     *
     * @param bytes byte array to load from
     * @return {@link NDArray}
     */
    default NDArray decode(byte[] bytes) {
        try (DataInputStream dis = new DataInputStream(new ByteArrayInputStream(bytes))) {
            return decode(dis);
        } catch (IOException e) {
            throw new IllegalArgumentException("NDArray decoding failed", e);
        }
    }

    /**
     * Decodes {@link NDArray} through {@link DataInputStream}.
     *
     * @param is input stream data to load from
     * @return {@link NDArray}
     * @throws IOException data is not readable
     */
    default NDArray decode(InputStream is) throws IOException {
        return NDSerializer.decode(this, is);
    }

    /**
     * Loads the NDArrays saved to a file.
     *
     * @param path the path to the file
     * @return the loaded arrays
     */
    NDList load(Path path);

    /**
     * Loads the NDArrays saved to a file.
     *
     * @param path the path to the file
     * @param device the device to use for the loaded arrays
     * @return the loaded arrays
     */
    default NDList load(Path path, Device device) {
        if (device == null || device.equals(getDevice())) {
            return load(path);
        }
        return newSubManager(device).load(path);
    }

    /**
     * Sets the name for the NDManager.
     *
     * @param name the name assigned to the manager
     */
    void setName(String name);

    /**
     * Gets the name of the NDManager.
     *
     * @return name
     */
    String getName();

    /**
     * Creates an instance of {@link NDArray} with specified {@link Shape} filled with zeros.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     * @see #zeros(Shape, DataType, Device)
     */
    default NDArray zeros(Shape shape) {
        return zeros(shape, DataType.FLOAT32);
    }

    /**
     * Creates an instance of {@link NDArray} with specified {@link Shape} filled with zeros.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     * @see #zeros(Shape, DataType, Device)
     */
    NDArray zeros(Shape shape, DataType dataType);

    /**
     * Creates an instance of {@link NDArray} with specified {@link Device}, {@link Shape}, and
     * {@link DataType} filled with zeros.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    default NDArray zeros(Shape shape, DataType dataType, Device device) {
        if (device == null || device.equals(getDevice())) {
            return zeros(shape, dataType);
        }
        return newSubManager(device).zeros(shape, dataType);
    }

    /**
     * Creates an instance of {@link NDArray} with specified {@link Shape} filled with ones.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    NDArray ones(Shape shape, DataType dataType);

    /**
     * Creates an instance of {@link NDArray} with specified {@link Shape} filled with ones.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    default NDArray ones(Shape shape) {
        return ones(shape, DataType.FLOAT32);
    }

    /**
     * Creates an instance of {@link NDArray} with specified {@link Device}, {@link Shape}, and
     * {@link DataType} filled with ones.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    default NDArray ones(Shape shape, DataType dataType, Device device) {
        if (device == null || device.equals(getDevice())) {
            return ones(shape, dataType);
        }
        return newSubManager(device).ones(shape, dataType);
    }

    /**
     * Return a new {@code NDArray} of given shape, filled with value.
     *
     * @param shape shape of a new {@code NDArray}
     * @param value fill value
     * @return {@code NDArray} of fill value with the given shape
     */
    default NDArray full(Shape shape, int value) {
        return full(shape, value, DataType.INT32);
    }

    /**
     * Return a new {@code NDArray} of given shape, filled with value.
     *
     * @param shape shape of a new {@code NDArray}
     * @param value fill value
     * @return {@code NDArray} of fill value with the given shape
     */
    default NDArray full(Shape shape, float value) {
        return full(shape, value, DataType.FLOAT32);
    }

    /**
     * Return a new {@code NDArray} of given shape, filled with value.
     *
     * @param shape shape of a new {@code NDArray}
     * @param value fill value
     * @param dataType the desired data-type for the {@link NDArray}
     * @return {@code NDArray} of fill value with the given shape
     */
    NDArray full(Shape shape, float value, DataType dataType);

    /**
     * Return a new {@code NDArray} of given shape, device, filled with value.
     *
     * @param shape shape of a new {@code NDArray}
     * @param value fill value
     * @param dataType the desired data-type for the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return {@code NDArray} of fill value with the given shape
     */
    default NDArray full(Shape shape, float value, DataType dataType, Device device) {
        if (device == null || device.equals(getDevice())) {
            return full(shape, value, dataType);
        }
        return newSubManager(device).full(shape, value, dataType);
    }

    /**
     * Returns evenly spaced values starting from 0.
     *
     * <p>Values are generated within the half-open interval [start, stop) (in other words, the
     * interval including start but excluding stop). For integer arguments, the function is
     * equivalent to the Python built-in range function, but returns an instance of {@link NDArray}
     * rather than a list.
     *
     * @param stop the end of the interval. The interval does not include this value
     * @return a new instance of {@link NDArray}
     */
    default NDArray arange(int stop) {
        return arange(0, stop, 1, DataType.INT32);
    }

    /**
     * Returns evenly spaced values starting from 0.
     *
     * <p>Values are generated within the half-open interval [start, stop) (in other words, the
     * interval including start but excluding stop). For integer arguments, the function is
     * equivalent to the Python built-in range function, but returns an instance of {@link NDArray}
     * rather than a list.
     *
     * @param stop the end of the interval. The interval does not include this value
     * @return a new instance of {@link NDArray}
     */
    default NDArray arange(float stop) {
        return arange(0.0f, stop, 1.0f, DataType.FLOAT32);
    }

    /**
     * Returns evenly spaced values within a given interval with step 1.
     *
     * <p>Values are generated within the half-open interval [start, stop) (in other words, the
     * interval including start but excluding stop). For integer arguments, the function is
     * equivalent to the Python built-in range function, but returns an instance of {@link NDArray}
     * rather than a list.
     *
     * @param start the start of interval. The interval includes this value
     * @param stop the end of interval. The interval does not include this value
     * @return a new instance of {@link NDArray}
     */
    default NDArray arange(int start, int stop) {
        return arange(start, stop, 1, DataType.INT32);
    }

    /**
     * Returns evenly spaced values within a given interval with step 1.
     *
     * <p>Values are generated within the half-open interval [start, stop) (in other words, the
     * interval including start but excluding stop). For integer arguments, the function is
     * equivalent to the Python built-in range function, but returns an instance of {@link NDArray}
     * rather than a list.
     *
     * @param start the start of interval. The interval includes this value
     * @param stop the end of interval. The interval does not include this value
     * @return a new instance of {@link NDArray}
     */
    default NDArray arange(float start, float stop) {
        return arange(start, stop, 1.0f, DataType.FLOAT32);
    }

    /**
     * Returns evenly spaced values within a given interval.
     *
     * <p>Values are generated within the half-open interval [start, stop) (in other words, the
     * interval including start but excluding stop). For integer arguments, the function is
     * equivalent to the Python built-in range function, but returns an instance of {@link NDArray}
     * rather than a list.
     *
     * @param start the start of interval. The interval includes this value
     * @param stop the end of interval. The interval does not include this value
     * @param step the spacing between values
     * @return a new instance of {@link NDArray}
     */
    default NDArray arange(int start, int stop, int step) {
        return arange(start, stop, step, DataType.INT32);
    }

    /**
     * Returns evenly spaced values within a given interval.
     *
     * <p>Values are generated within the half-open interval [start, stop) (in other words, the
     * interval including start but excluding stop). For integer arguments, the function is
     * equivalent to the Python built-in range function, but returns an instance of {@link NDArray}
     * rather than a list.
     *
     * @param start the start of interval. The interval includes this value
     * @param stop the end of interval. The interval does not include this value
     * @param step the spacing between values
     * @return a new instance of {@link NDArray}
     */
    default NDArray arange(float start, float stop, float step) {
        return arange(start, stop, step, DataType.FLOAT32);
    }

    /**
     * Returns evenly spaced values within a given interval.
     *
     * <p>Values are generated within the half-open interval [start, stop) (in other words, the
     * interval including start but excluding stop). For integer arguments, the function is
     * equivalent to the Python built-in range function, but returns an instance of {@link NDArray}
     * rather than a list.
     *
     * @param start the start of interval. The interval includes this value
     * @param stop the end of interval. The interval does not include this value
     * @param step the spacing between values
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    default NDArray arange(int start, int stop, int step, DataType dataType) {
        return arange((float) start, (float) stop, (float) step, dataType);
    }

    /**
     * Returns evenly spaced values within a given interval.
     *
     * <p>Values are generated within the half-open interval [start, stop) (in other words, the
     * interval including start but excluding stop). For integer arguments, the function is
     * equivalent to the Python built-in range function, but returns an instance of {@link NDArray}
     * rather than a list.
     *
     * @param start the start of interval. The interval includes this value
     * @param stop the end of interval. The interval does not include this value
     * @param step the spacing between values
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    NDArray arange(float start, float stop, float step, DataType dataType);

    /**
     * Returns evenly spaced values within a given interval.
     *
     * <p>Values are generated within the half-open interval [start, stop) (in other words, the
     * interval including start but excluding stop). For integer arguments, the function is
     * equivalent to the Python built-in range function, but returns an instance of {@link NDArray}
     * rather than a list.
     *
     * @param start the start of interval. The interval includes this value
     * @param stop the end of interval. The interval does not include this value
     * @param step the spacing between values
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    default NDArray arange(float start, float stop, float step, DataType dataType, Device device) {
        if (device == null || device.equals(getDevice())) {
            return arange(start, stop, step, dataType);
        }
        return newSubManager(device).arange(start, stop, step, dataType);
    }

    /**
     * Returns a 2-D array with ones on the diagonal and zeros elsewhere.
     *
     * @param rows the number of rows and cols in the output
     * @return a {@link NDArray} where all elements are equal to zero, except for the k-th diagonal,
     *     whose values are equal to one
     */
    default NDArray eye(int rows) {
        return eye(rows, rows, 0, DataType.FLOAT32);
    }

    /**
     * Returns a 2-D array with ones on the diagonal and zeros elsewhere.
     *
     * @param rows the number of rows and cols in the output
     * @param k the index of the diagonal: a positive value refers to an upper diagonal, and a
     *     negative value to a lower diagonal
     * @return a {@link NDArray} where all elements are equal to zero, except for the k-th diagonal,
     *     whose values are equal to one
     */
    default NDArray eye(int rows, int k) {
        return eye(rows, rows, k, DataType.FLOAT32);
    }

    /**
     * Returns a 2-D array with ones on the diagonal and zeros elsewhere.
     *
     * @param rows the number of rows in the output
     * @param cols the number of columns in the output
     * @param k the index of the diagonal: a positive value refers to an upper diagonal, and a
     *     negative value to a lower diagonal
     * @return a {@link NDArray} where all elements are equal to zero, except for the k-th diagonal,
     *     whose values are equal to one
     */
    default NDArray eye(int rows, int cols, int k) {
        return eye(rows, cols, k, DataType.FLOAT32);
    }

    /**
     * Returns a 2-D array with ones on the diagonal and zeros elsewhere.
     *
     * @param rows the number of rows int the output
     * @param cols the number of columns in the output
     * @param k the index of the diagonal: a positive value refers to an upper diagonal, and a
     *     negative value to a lower diagonal
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @return a {@link NDArray} where all elements are equal to zero, except for the k-th diagonal,
     *     whose values are equal to one
     */
    NDArray eye(int rows, int cols, int k, DataType dataType);

    /**
     * Returns a 2-D array with ones on the diagonal and zeros elsewhere.
     *
     * @param rows the number of rows int the output
     * @param cols the number of columns in the output
     * @param k the index of the diagonal: a positive value refers to an upper diagonal, and a
     *     negative value to a lower diagonal
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return a {@link NDArray} where all elements are equal to zero, except for the k-th diagonal,
     *     whose values are equal to one
     */
    default NDArray eye(int rows, int cols, int k, DataType dataType, Device device) {
        if (device == null || device.equals(getDevice())) {
            return eye(rows, cols, k, dataType);
        }
        return newSubManager(device).eye(rows, cols, k, dataType);
    }

    /**
     * Returns evenly spaced numbers over a specified interval.
     *
     * <p>Returns num evenly spaced samples, calculated over the interval [start, stop].
     *
     * @param start the starting value of the sequence
     * @param stop the end value of the sequence
     * @param num the number of samples to generate
     * @return a new instance of {@link NDArray}
     */
    default NDArray linspace(int start, int stop, int num) {
        return linspace(start, stop, num, true);
    }

    /**
     * Returns evenly spaced numbers over a specified interval.
     *
     * <p>Returns num evenly spaced samples, calculated over the interval [start, stop].
     *
     * @param start the starting value of the sequence
     * @param stop the end value of the sequence
     * @param num the number of samples to generate
     * @return a new instance of {@link NDArray}
     */
    default NDArray linspace(float start, float stop, int num) {
        return linspace(start, stop, num, true);
    }

    /**
     * Returns evenly spaced numbers over a specified interval.
     *
     * <p>Returns num evenly spaced samples, calculated over the interval [start, stop].The endpoint
     * of the interval can optionally be excluded.
     *
     * @param start the starting value of the sequence
     * @param stop the end value of the sequence
     * @param num the number of samples to generate
     * @param endpoint if {@code true}, stop is the last sample, otherwise, it is not included
     * @return a new instance of {@link NDArray}
     */
    default NDArray linspace(int start, int stop, int num, boolean endpoint) {
        return linspace((float) start, (float) stop, num, endpoint);
    }

    /**
     * Returns evenly spaced numbers over a specified interval.
     *
     * <p>Returns num evenly spaced samples, calculated over the interval [start, stop].The endpoint
     * of the interval can optionally be excluded.
     *
     * @param start the starting value of the sequence
     * @param stop the end value of the sequence
     * @param num the number of samples to generate
     * @param endpoint if {@code true}, stop is the last sample, otherwise, it is not included
     * @return a new instance of {@link NDArray}
     */
    NDArray linspace(float start, float stop, int num, boolean endpoint);

    /**
     * Returns evenly spaced numbers over a specified interval.
     *
     * <p>Returns num evenly spaced samples, calculated over the interval [start, stop].The endpoint
     * of the interval can optionally be excluded.
     *
     * @param start the starting value of the sequence
     * @param stop the end value of the sequence
     * @param num the number of samples to generate
     * @param endpoint if {@code true}, stop is the last sample, otherwise, it is not included
     * @param device the {@link Device} of the {@link NDArray}
     * @return a new instance of {@link NDArray}
     */
    default NDArray linspace(float start, float stop, int num, boolean endpoint, Device device) {
        if (device == null || device.equals(getDevice())) {
            return linspace(start, stop, num, endpoint);
        }
        return newSubManager(device).linspace(start, stop, num, endpoint);
    }

    /**
     * Returns random integer values from low (inclusive) to high (exclusive).
     *
     * @param low Lowest (signed) longs to be drawn from the distribution
     * @param high one above the largest (signed) long to be drawn from the distribution
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @return the drawn samples {@link NDArray}
     */
    NDArray randomInteger(long low, long high, Shape shape, DataType dataType);

    /**
     * Draws samples from a uniform distribution.
     *
     * <p>Samples are uniformly distributed over the half-open interval [low, high) (includes low,
     * but excludes high). In other words, any value within the given interval is equally likely to
     * be drawn by uniform.
     *
     * @param low the lower boundary of the output interval. All values generated will be greater
     *     than or equal to low.
     * @param high the upper boundary of the output interval. All values generated will be less than
     *     high.
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return the drawn samples {@link NDArray}
     */
    default NDArray randomUniform(float low, float high, Shape shape) {
        return randomUniform(low, high, shape, DataType.FLOAT32);
    }

    /**
     * Draws samples from a uniform distribution.
     *
     * <p>Samples are uniformly distributed over the half-open interval [low, high) (includes low,
     * but excludes high). In other words, any value within the given interval is equally likely to
     * be drawn by uniform.
     *
     * @param low the lower boundary of the output interval. All values generated will be greater
     *     than or equal to low.
     * @param high the upper boundary of the output interval. All values generated will be less than
     *     high.
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @return the drawn samples {@link NDArray}
     */
    NDArray randomUniform(float low, float high, Shape shape, DataType dataType);

    /**
     * Draws samples from a uniform distribution.
     *
     * <p>Samples are uniformly distributed over the half-open interval [low, high) (includes low,
     * but excludes high). In other words, any value within the given interval is equally likely to
     * be drawn by uniform.
     *
     * @param low the lower boundary of the output interval. All values generated will be greater
     *     than or equal to low.
     * @param high the upper boundary of the output interval. All values generated will be less than
     *     high.
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return the drawn samples {@link NDArray}
     */
    default NDArray randomUniform(
            float low, float high, Shape shape, DataType dataType, Device device) {
        if (device == null || device.equals(getDevice())) {
            return randomUniform(low, high, shape, dataType);
        }
        return newSubManager(device).randomUniform(low, high, shape, dataType);
    }

    /**
     * Draws random samples from a normal (Gaussian) distribution with mean 0 and standard deviation
     * 1.
     *
     * <p>Samples are distributed according to a normal distribution parametrized by mean = 0 and
     * standard deviation = 1.
     *
     * @param shape the output {@link Shape}
     * @return the drawn samples {@link NDArray}
     */
    default NDArray randomNormal(Shape shape) {
        return randomNormal(0f, 1f, shape, DataType.FLOAT32);
    }

    /**
     * Draws random samples from a normal (Gaussian) distribution with mean 0 and standard deviation
     * 1.
     *
     * @param shape the output {@link Shape}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @return the drawn samples {@link NDArray}
     */
    default NDArray randomNormal(Shape shape, DataType dataType) {
        return randomNormal(0.0f, 1.0f, shape, dataType);
    }

    /**
     * Draws random samples from a normal (Gaussian) distribution.
     *
     * @param loc the mean (centre) of the distribution
     * @param scale the standard deviation (spread or "width") of the distribution
     * @param shape the output {@link Shape}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @return the drawn samples {@link NDArray}
     */
    NDArray randomNormal(float loc, float scale, Shape shape, DataType dataType);

    /**
     * Draws random samples from a normal (Gaussian) distribution.
     *
     * @param loc the mean (centre) of the distribution
     * @param scale the standard deviation (spread or "width") of the distribution
     * @param shape the output {@link Shape}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return the drawn samples {@link NDArray}
     */
    default NDArray randomNormal(
            float loc, float scale, Shape shape, DataType dataType, Device device) {
        if (device == null || device.equals(getDevice())) {
            return randomNormal(loc, scale, shape, dataType);
        }
        return newSubManager(device).randomNormal(loc, scale, shape, dataType);
    }

    /**
     * Draws random samples from a normal (Gaussian) distribution with mean 0 and standard deviation
     * 1, discarding and re-drawing any samples that are more than two standard deviations from the
     * mean.
     *
     * <p>Samples are distributed according to a normal distribution parametrized by mean = 0 and
     * standard deviation = 1.
     *
     * @param shape the output {@link Shape}
     * @return the drawn samples {@link NDArray}
     */
    default NDArray truncatedNormal(Shape shape) {
        return truncatedNormal(0f, 1f, shape, DataType.FLOAT32);
    }

    /**
     * Draws random samples from a normal (Gaussian) distribution with mean 0 and standard deviation
     * 1, discarding and re-drawing any samples that are more than two standard deviations from the
     * mean.
     *
     * @param shape the output {@link Shape}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @return the drawn samples {@link NDArray}
     */
    default NDArray truncatedNormal(Shape shape, DataType dataType) {
        return truncatedNormal(0.0f, 1.0f, shape, dataType);
    }

    /**
     * Draws random samples from a normal (Gaussian) distribution, discarding and re-drawing any
     * samples that are more than two standard deviations from the mean.
     *
     * @param loc the mean (centre) of the distribution
     * @param scale the standard deviation (spread or "width") of the distribution
     * @param shape the output {@link Shape}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @return the drawn samples {@link NDArray}
     */
    NDArray truncatedNormal(float loc, float scale, Shape shape, DataType dataType);

    /**
     * Draws random samples from a normal (Gaussian) distribution, discarding and re-drawing any
     * samples that are more than two standard deviations from the mean.
     *
     * @param loc the mean (centre) of the distribution
     * @param scale the standard deviation (spread or "width") of the distribution
     * @param shape the output {@link Shape}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return the drawn samples {@link NDArray}
     */
    default NDArray truncatedNormal(
            float loc, float scale, Shape shape, DataType dataType, Device device) {
        if (device == null || device.equals(getDevice())) {
            return truncatedNormal(loc, scale, shape, dataType);
        }
        return newSubManager(device).truncatedNormal(loc, scale, shape, dataType);
    }

    /**
     * Draw samples from a multinomial distribution.
     *
     * <p>The multinomial distribution is a multivariate generalization of the binomial
     * distribution. Take an experiment with one of p possible outcomes. An example of such an
     * experiment is throwing a dice, where the outcome can be 1 through 6. Each sample drawn from
     * the distribution represents n such experiments. Its values, X_i = [X_0, X_1, ..., X_p],
     * represent the number of times the outcome was i.
     *
     * @param n the number of experiments
     * @param pValues the probabilities of each of the p different outcomes. These should sum to 1
     *     The last element is always assumed to account for the remaining probability, as long as
     *     pValues.sum().getFloat() &lt;= 1)
     * @return the drawn samples {@link NDArray}
     */
    NDArray randomMultinomial(int n, NDArray pValues);

    /**
     * Draw samples from a multinomial distribution.
     *
     * <p>The multinomial distribution is a multivariate generalization of the binomial
     * distribution. Take an experiment with one of p possible outcomes. An example of such an
     * experiment is throwing a dice, where the outcome can be 1 through 6. Each sample drawn from
     * the distribution represents n such experiments. Its values, X_i = [X_0, X_1, ..., X_p],
     * represent the number of times the outcome was i.
     *
     * @param n the number of experiments
     * @param pValues the probabilities of each of the p different outcomes. These should sum to 1
     *     The last element is always assumed to account for the remaining probability, as long as
     *     pValues.sum().getFloat() &lt;= 1)
     * @param shape the output {@link Shape}
     * @return the drawn samples {@link NDArray}
     */
    NDArray randomMultinomial(int n, NDArray pValues, Shape shape);

    /**
     * Check if the manager is still valid.
     *
     * @return the current status
     */
    boolean isOpen();

    /**
     * Returns the parent {@code NDManager}.
     *
     * @return the parent {@code NDManager}
     */
    NDManager getParentManager();

    /**
     * Creates a child {@code NDManager}.
     *
     * <p>Child {@code NDManager} will inherit default {@link Device} from this {@code NDManager}.
     *
     * @return a child {@code NDManager}
     */
    NDManager newSubManager();

    /**
     * Creates a child {@code NDManager} with specified default {@link Device}.
     *
     * @param device the default {@link Device}
     * @return a child {@code NDManager}
     */
    NDManager newSubManager(Device device);

    /**
     * Returns the default {@link Device} of this {@code NDManager}.
     *
     * @return the default {@link Device} of this {@code NDManager}
     */
    Device getDevice();

    /**
     * Attaches a resource to this {@code NDManager}.
     *
     * <p>The attached resource will be closed when this {@code NDManager} is closed.
     *
     * <p>This attachment is internal. Many resources will internally track which manager they are
     * attached to. In that case, you should call {@link NDResource#attach(NDManager)} instead and
     * that should then call attachInternal.
     *
     * @param resourceId the unique resourceId
     * @param resource the {@link AutoCloseable} resource to be attached
     */
    void attachInternal(String resourceId, AutoCloseable resource);

    /**
     * Temporarily attaches a resource to this {@code NDManager} to be returned when this is closed.
     *
     * <p>The attached resource will be returned to it's original manager when this {@code
     * NDManager} is closed.
     *
     * <p>This attachment is internal. Many resources will internally track which manager they are
     * attached to. In that case, you should call {@link NDResource#attach(NDManager)} instead and
     * that should then call tempAttachInternal.
     *
     * @param originalManager the original manager to return the resource to
     * @param resourceId the unique resourceId
     * @param resource the {@link AutoCloseable} resource to be attached
     */
    void tempAttachInternal(NDManager originalManager, String resourceId, NDResource resource);

    /**
     * Detaches a {@link NDArray} from this {@code NDManager}'s lifecycle.
     *
     * <p>The detached {@link NDArray} become un-managed, it's user's responsibility to close the
     * resource. Failed to close the resource has to wait on GC to be freed, and might cause out of
     * native memory.
     *
     * <p>This detach is internal. Many resources will internally track which manager they are
     * attached to. In that case, you should call {@link NDResource#detach()} instead and that
     * should then call detachInternal.
     *
     * @param resourceId the resourceId to be removed from this {@code NDManager}'s lifecycle
     */
    void detachInternal(String resourceId);

    /**
     * Returns a value outside of this manager by attaching to this manager's parent.
     *
     * @param resource the resource to return
     * @param <T> the type of the resource
     * @return the passed in resource, after attaching to a new manager
     */
    default <T extends NDResource> T ret(T resource) {
        resource.attach(getParentManager());
        return resource;
    }

    /**
     * Attaches all resources to this manager.
     *
     * @param resources the resources to attach
     * @see NDResource#attach(NDManager)
     */
    default void attachAll(NDResource... resources) {
        for (NDResource resource : resources) {
            resource.attach(this);
        }
    }

    /**
     * Temporarily attaches all resources to this manager.
     *
     * @param resources the resources to attach
     * @see NDResource#tempAttach(NDManager)
     */
    default void tempAttachAll(NDResource... resources) {
        for (NDResource resource : resources) {
            resource.tempAttach(this);
        }
    }

    /**
     * An engine specific generic invocation to native operation.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause a portability issue. Native operation may not be compatible between
     * each version.
     *
     * @param operation the native operation to perform
     * @param src the {@link NDList} of source {@link NDArray}
     * @param dest the {@link NDList} to save output to
     * @param params the parameters to be passed to the native operation
     * @throws IllegalArgumentException if operation is not supported by Engine
     * @throws EngineException if operation failed in native engine
     */
    void invoke(String operation, NDArray[] src, NDArray[] dest, PairList<String, ?> params);

    /**
     * An engine specific generic invocation to native operation.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause a portability issue. Native operation may not compatible between
     * each version.
     *
     * @param operation the native operation to perform
     * @param src the {@link NDList} of source {@link NDArray}
     * @param params the parameters to be passed to the native operation
     * @return the output array of {@link NDArray}
     * @throws IllegalArgumentException if operation is not supported by Engine
     * @throws EngineException if operation failed in native engine
     */
    NDList invoke(String operation, NDList src, PairList<String, ?> params);

    /**
     * Returns the {@link Engine} associated with this manager.
     *
     * @return the {@link Engine} associated with this manager
     */
    Engine getEngine();

    /** {@inheritDoc} */
    @Override
    void close();
}

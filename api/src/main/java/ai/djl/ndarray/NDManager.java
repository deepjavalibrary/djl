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
import ai.djl.ndarray.types.DataDesc;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.PairList;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.file.Path;

/**
 * NDArray managers are used to create <I>NDArrays</I> (n-dimensional array on native engine).
 *
 * <p>NDManager is implemented in each deep learning framework {@link Engine}. {@link NDArray}s are
 * resources that allocated in each deep learning framework's native memory space. NDManager is the
 * key class that manages those native resources.
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
 *         NDArray array = <b>manager</b>.create(dataDesc);
 *         array.set(input);
 *         return new NDList(array);
 *     } // NDArrays created in this method will be closed after method return.
 * }
 * </pre>
 *
 * <p>NDManager has a hierarchical structure; it has a single parent NDManager and has child
 * NDManagers. When the parent NDManager is closed, all children will be closed as well.
 *
 * <p>The Joule framework manage NDManager's lifecycle by default. You only need to manage the user
 * created child NDManager. Child NDManager becomes useful when you create a large amount of
 * temporary NDArrays and want to free the resources earlier than parent NDManager's lifecycle.
 *
 * <p>The following is an example of such a use case:
 *
 * <pre>
 * public class MyTranslator implements Translator&lt;List&lt;FloatBuffer&gt;&gt;, String&gt; {
 *
 *     &#064;Override
 *     public NDList processInput(TranslatorContext ctx, List&lt;FloatBuffer&gt; input) {
 *         NDManager manager = ctx.getNDManager();
 *         NDArray array = manager.create(dataDesc);
 *         for (int i = 0; i &lt; input.size(); ++i) {
 *             try (<b>NDManager childManager = manager.newSubManager()</b>) {
 *                  NDArray tmp = <b>childManager</b>.create(itemDataDesc);
 *                  tmp.put(input.get(i);
 *                  array.put(i, tmp);
 *             } // NDArray <i>tmp</i> will be closed here
 *         }
 *         return new NDList(array);
 *     }
 * }
 * </pre>
 *
 * <p>You can also close an individual NDArray. NDManager won't double close them. In certain use
 * cases, you might want to return an NDArray outside of NDManager's scope.
 *
 * @see NDArray
 * @see Translator
 * @see TranslatorContext#getNDManager()
 */
public interface NDManager extends AutoCloseable {

    /**
     * Creates a new top-level {@code NDManager}.
     *
     * <p>{@code NDManager} will inherit default {@link Device}.
     *
     * @return Returns a new top-level {@code NDManager}
     */
    static NDManager newBaseManager() {
        return Engine.getInstance().newBaseManager();
    }

    /**
     * Creates a new top-level {@code NDManager} with specified {@link Device}.
     *
     * @param device default {@link Device}
     * @return Returns a new top-level {@code NDManager}
     */
    static NDManager newBaseManager(Device device) {
        return Engine.getInstance().newBaseManager(device);
    }

    /**
     * Allocates a new engine specific direct byte buffer.
     *
     * @param capacity The new buffer's capacity, in bytes
     * @return The new byte buffer
     */
    ByteBuffer allocateDirect(int capacity);

    /**
     * Creates an uninitialized instance of {@link DataType#FLOAT32} {@link NDArray} with specified
     * {@link Shape}.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    default NDArray create(Shape shape) {
        return create(shape, DataType.FLOAT32, getDevice());
    }

    /**
     * Creates and initializes a scalar {@link NDArray}.
     *
     * @param data the float that needs to be set
     * @return new instance of {@link NDArray}
     */
    default NDArray create(float data) {
        return create(new float[] {data}, new Shape());
    }

    /**
     * Creates and initializes a scalar {@link NDArray}.
     *
     * @param data the float data that needs to be set
     * @return new instance of {@link NDArray}
     */
    default NDArray create(int data) {
        return create(new int[] {data}, new Shape());
    }

    /**
     * Creates and initializes a scalar {@link NDArray}.
     *
     * @param data the double data that needs to be set
     * @return new instance of {@link NDArray}
     */
    default NDArray create(double data) {
        return create(new double[] {data}, new Shape());
    }

    /**
     * Creates and initializes a scalar {@link NDArray}.
     *
     * @param data the long data that needs to be set
     * @return new instance of {@link NDArray}
     */
    default NDArray create(long data) {
        return create(new long[] {data}, new Shape());
    }

    /**
     * Creates and initializes a scalar {@link NDArray}.
     *
     * @param data the byte data that needs to be set
     * @return new instance of {@link NDArray}
     */
    default NDArray create(byte data) {
        return create(new byte[] {data}, new Shape());
    }

    /**
     * Creates and initializes a 1D {@link NDArray}.
     *
     * @param data the float array that needs to be set
     * @return new instance of {@link NDArray}
     */
    default NDArray create(float[] data) {
        return create(data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 1D {@link NDArray}.
     *
     * @param data the float array that needs to be set
     * @return new instance of {@link NDArray}
     */
    default NDArray create(int[] data) {
        return create(data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 1D {@link NDArray}.
     *
     * @param data the float array that needs to be set
     * @return new instance of {@link NDArray}
     */
    default NDArray create(double[] data) {
        return create(data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 1D {@link NDArray}.
     *
     * @param data the float array that needs to be set
     * @return new instance of {@link NDArray}
     */
    default NDArray create(long[] data) {
        return create(data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 1D {@link NDArray}.
     *
     * @param data the float array that needs to be set
     * @return new instance of {@link NDArray}
     */
    default NDArray create(byte[] data) {
        return create(data, new Shape(data.length));
    }

    /**
     * Creates and initializes a 2D {@link NDArray}.
     *
     * @param data the float array that needs to be set
     * @return new instance of {@link NDArray}
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
     * @return new instance of {@link NDArray}
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
     * @return new instance of {@link NDArray}
     */
    default NDArray create(double[][] data) {
        DoubleBuffer buffer = DoubleBuffer.allocate(data.length * data[0].length);
        for (double[] d : data) {
            buffer.put(d);
        }
        // TODO Check this. It may not work with Java 11
        buffer.rewind();
        return create(buffer, new Shape(data.length, data[0].length));
    }

    /**
     * Creates and initializes a 2D {@link NDArray}.
     *
     * @param data the float array that needs to be set
     * @return new instance of {@link NDArray}
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
     * Creates and initializes a 2D {@link NDArray}.
     *
     * @param data the float array that needs to be set
     * @return new instance of {@link NDArray}
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
     * Creates and initializes a {@link NDArray} with specified {@link Shape}.
     *
     * <p>{@link DataType} of the NDArray will determined by type of Buffer.
     *
     * @param data data to initialize the {@code NDArray}
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    default NDArray create(Buffer data, Shape shape) {
        DataType dataType = DataType.fromBuffer(data);
        NDArray array = create(shape, dataType, getDevice());
        array.set(data);
        return array;
    }

    /**
     * Creates an uninitialized instance of {@link NDArray} with specified {@link Shape}, and {@link
     * DataType}.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    default NDArray create(Shape shape, DataType dataType) {
        return create(shape, dataType, getDevice());
    }

    /**
     * Creates an instance of {@link NDArray} with specified {@link DataDesc}.
     *
     * @param dataDesc the {@link DataDesc} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    default NDArray create(DataDesc dataDesc) {
        return create(dataDesc.getShape(), dataDesc.getDataType(), getDevice());
    }

    /**
     * Creates and initialize an instance of {@link NDArray} with specified {@link DataDesc}.
     *
     * @param data data to initialize the {@code NDArray}
     * @param dataDesc the {@link DataDesc} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    default NDArray create(Buffer data, DataDesc dataDesc) {
        NDArray array = create(dataDesc);
        array.set(data);
        return array;
    }

    /**
     * Creates and initialize an instance of {@link NDArray} with specified {@link Shape} and float
     * array.
     *
     * @param data the float array that needs to be set
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    default NDArray create(float[] data, Shape shape) {
        return create(FloatBuffer.wrap(data), shape);
    }

    /**
     * Creates and initialize an instance of {@link NDArray} with specified {@link Shape} and int
     * array.
     *
     * @param data the float array that needs to be set
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    default NDArray create(int[] data, Shape shape) {
        return create(IntBuffer.wrap(data), shape);
    }

    /**
     * Creates and initialize an instance of {@link NDArray} with specified {@link Shape} and double
     * array.
     *
     * @param data the float array that needs to be set
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    default NDArray create(double[] data, Shape shape) {
        return create(DoubleBuffer.wrap(data), shape);
    }

    /**
     * Creates and initialize an instance of {@link NDArray} with specified {@link Shape} and long
     * array.
     *
     * @param data the float array that needs to be set
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    default NDArray create(long[] data, Shape shape) {
        return create(LongBuffer.wrap(data), shape);
    }

    /**
     * Creates and initialize an instance of {@link NDArray} with specified {@link Shape} and byte
     * array.
     *
     * @param data the float array that needs to be set
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    default NDArray create(byte[] data, Shape shape) {
        return create(ByteBuffer.wrap(data), shape);
    }

    /**
     * Creates an uninitialized instance of {@link NDArray} with specified {@link Shape}, {@link
     * DataType} and {@link Device}.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    NDArray create(Shape shape, DataType dataType, Device device);

    /**
     * Create a Compressed Sparse Row Storage (CSR) Format Matrix.
     *
     * @param data the data to set for the CSR Matrix
     * @param indptr The indptr array is what will help identify the rows where the data appears
     * @param indices The indices array stores the column index for each non-zero element in data
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    NDArray createCSR(Buffer data, long[] indptr, long[] indices, Shape shape, Device device);

    /**
     * Create a Compressed Sparse Row Storage (CSR) Format Matrix.
     *
     * @param data the data to set for the CSR Matrix
     * @param indptr The indptr array is what will help identify the rows where the data appears
     * @param indices The indices array stores the column index for each non-zero element in data
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    default NDArray createCSR(Buffer data, long[] indptr, long[] indices, Shape shape) {
        return createCSR(data, indptr, indices, shape, getDevice());
    }

    /**
     * RowSparseNDArray stores the matrix in row sparse format.
     *
     * @param data the data to set for the Row Sparse NDArray
     * @param dataShape the {@link Shape} of the data {@link NDArray}
     * @param indices the indices to store the data
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    NDArray createRowSparse(
            Buffer data, Shape dataShape, long[] indices, Shape shape, Device device);

    /**
     * RowSparseNDArray stores the matrix in row sparse format.
     *
     * @param data the data to set for the Row Sparse NDArray
     * @param dataShape the {@link Shape} of the data {@link NDArray}
     * @param indices the indices to store the data
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    default NDArray createRowSparse(Buffer data, Shape dataShape, long[] indices, Shape shape) {
        return createRowSparse(data, dataShape, indices, shape, getDevice());
    }

    /**
     * Load NDArrays from file.
     *
     * @param path path to load from
     * @return list of NDArray. Tags might be present or be null.
     */
    NDList load(Path path);

    /**
     * Save NDArray to file.
     *
     * @param path path save to
     * @param ndList list of NDArray to save
     */
    void save(Path path, NDList ndList);

    /**
     * Creates an instance of {@link NDArray} with specified {@link Shape} filled with zeros.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     * @see #zeros(Shape, DataType, Device)
     */
    default NDArray zeros(Shape shape) {
        return zeros(shape, DataType.FLOAT32, getDevice());
    }

    /**
     * Creates an instance of {@link NDArray} with specified {@link DataDesc} filled with zeros.
     *
     * @param dataDesc the {@link DataDesc} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    default NDArray zeros(DataDesc dataDesc) {
        return zeros(dataDesc.getShape(), dataDesc.getDataType(), getDevice());
    }

    /**
     * Creates an instance of {@link NDArray} with specified {@link Device}, * {@link Shape}, and
     * {@link DataType} filled with zeros.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    NDArray zeros(Shape shape, DataType dataType, Device device);

    /**
     * Creates an instance of {@link NDArray} with specified {@link Shape} filled with ones.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    default NDArray ones(Shape shape) {
        return ones(shape, DataType.FLOAT32, getDevice());
    }

    /**
     * Creates an instance of {@link NDArray} with specified {@link DataDesc} filled with ones.
     *
     * @param dataDesc the {@link DataDesc} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    default NDArray ones(DataDesc dataDesc) {
        return ones(dataDesc.getShape(), dataDesc.getDataType(), getDevice());
    }

    /**
     * Creates an instance of {@link NDArray} with specified {@link Device}, {@link Shape}, and
     * {@link DataType} filled with ones.
     *
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    NDArray ones(Shape shape, DataType dataType, Device device);

    /**
     * Returns evenly spaced values starting from 0 on current device.
     *
     * <p>Values are generated within the half-open interval ``[start, stop)`` (in other words, the
     * interval including `start` but excluding `stop`). For integer arguments the function is
     * equivalent to the Python built-in `range` function, but returns an instance of {@link
     * NDArray} rather than a list.
     *
     * @param stop end of interval. The interval does not include this value.
     * @return new instance of {@link NDArray}
     */
    default NDArray arange(int stop) {
        return arange(0, stop, 1, DataType.FLOAT32, getDevice());
    }

    /**
     * Returns evenly spaced values within a given interval on current device with step 1.
     *
     * <p>Values are generated within the half-open interval ``[start, stop)`` (in other words, the
     * interval including `start` but excluding `stop`). For integer arguments the function is
     * equivalent to the Python built-in `range` function, but returns an instance of {@link
     * NDArray} rather than a list.
     *
     * @param start start of interval. The interval includes this value.
     * @param stop end of interval. The interval does not include this value.
     * @return new instance of {@link NDArray}
     */
    default NDArray arange(int start, int stop) {
        return arange(start, stop, 1, DataType.FLOAT32, getDevice());
    }

    /**
     * Returns evenly spaced values within a given interval on current device.
     *
     * <p>Values are generated within the half-open interval ``[start, stop)`` (in other words, the
     * interval including `start` but excluding `stop`). For integer arguments the function is
     * equivalent to the Python built-in `range` function, but returns an instance of {@link
     * NDArray} rather than a list.
     *
     * @param start start of interval. The interval includes this value.
     * @param stop end of interval. The interval does not include this value.
     * @param step spacing between values.
     * @return new instance of {@link NDArray}
     */
    default NDArray arange(int start, int stop, int step) {
        return arange(start, stop, step, DataType.FLOAT32, getDevice());
    }

    /**
     * Returns evenly spaced values within a given interval.
     *
     * <p>Values are generated within the half-open interval ``[start, stop)`` (in other words, the
     * interval including `start` but excluding `stop`). For integer arguments the function is
     * equivalent to the Python built-in `range` function, but returns an instance of {@link
     * NDArray} rather than a list.
     *
     * @param start start of interval, inclusive
     * @param stop end of interval, exclusive
     * @param step spacing between values
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    NDArray arange(int start, int stop, int step, DataType dataType, Device device);

    /**
     * Return a 2-D array with ones on the diagonal and zeros elsewhere on current device.
     *
     * @param rows number of rows and cols in the output
     * @return a {@link NDArray} where all elements are equal to zero, except for the k-th diagonal,
     *     whose values are equal to one.
     */
    default NDArray eye(int rows) {
        return eye(rows, rows, 0, DataType.FLOAT32, getDevice());
    }

    /**
     * Return a 2-D array with ones on the diagonal and zeros elsewhere on current device.
     *
     * @param rows number of rows and cols in the output
     * @param k index of the diagonal: 0 (the default) refers to the main diagonal, a positive value
     *     refers to an upper diagonal, and a negative value to a lower diagonal.
     * @return a {@link NDArray} where all elements are equal to zero, except for the k-th diagonal,
     *     whose values are equal to one.
     */
    default NDArray eye(int rows, int k) {
        return eye(rows, rows, k, DataType.FLOAT32, getDevice());
    }

    /**
     * Return a 2-D array with ones on the diagonal and zeros elsewhere on current device.
     *
     * @param rows number of rows in the output
     * @param cols number of columns in the output
     * @param k index of the diagonal: 0 (the default) refers to the main diagonal, a positive value
     *     refers to an upper diagonal, and a negative value to a lower diagonal.
     * @return a {@link NDArray} where all elements are equal to zero, except for the k-th diagonal,
     *     whose values are equal to one.
     */
    default NDArray eye(int rows, int cols, int k) {
        return eye(rows, cols, k, DataType.FLOAT32, getDevice());
    }

    /**
     * Return a 2-D array with ones on the diagonal and zeros elsewhere.
     *
     * @param rows number of rows int the output
     * @param cols number of columns in the output
     * @param k index of the diagonal: 0 (the default) refers to the main diagonal, a positive value
     *     refers to an upper diagonal, and a negative value to a lower diagonal.
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return a {@link NDArray} where all elements are equal to zero, except for the k-th diagonal,
     *     whose values are equal to one.
     */
    NDArray eye(int rows, int cols, int k, DataType dataType, Device device);

    /**
     * Return evenly spaced numbers over a specified interval on current device.
     *
     * <p>Returns num evenly spaced samples, calculated over the interval [start, stop].
     *
     * @param start the starting value of the sequence
     * @param stop the end value of the sequence
     * @param num number of samples to generate
     * @return new instance of {@link NDArray}
     */
    default NDArray linspace(double start, double stop, int num) {
        return linspace(start, stop, num, true, getDevice());
    }

    /**
     * Return evenly spaced numbers over a specified interval.
     *
     * <p>Returns num evenly spaced samples, calculated over the interval [start, stop].The endpoint
     * of the interval can optionally be excluded.
     *
     * @param start the starting value of the sequence
     * @param stop the end value of the sequence
     * @param num number of samples to generate
     * @param endpoint if {@code true}, stop is the last sample, otherwise, it is not included
     * @param device the {@link Device} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    NDArray linspace(double start, double stop, int num, boolean endpoint, Device device);

    /**
     * Draw random samples from a normal (Gaussian) distribution on current device.
     *
     * <p>Samples are uniformly distributed over the half-open interval ``[low, high)`` (includes
     * low, but excludes high). In other words, any value within the given interval is equally
     * likely to be drawn by `uniform`
     *
     * @param low Lower boundary of the output interval. All values generated will be greater than
     *     or equal to low.
     * @param high Upper boundary of the output interval. All values generated will be less than
     *     high.
     * @param shape the {@link Shape} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    default NDArray randomUniform(double low, double high, Shape shape) {
        return randomUniform(low, high, shape, DataType.FLOAT32, getDevice());
    }

    /**
     * Draw random samples from a normal (Gaussian) distribution.
     *
     * <p>Samples are uniformly distributed over the half-open interval ``[low, high)`` (includes
     * low, but excludes high). In other words, any value within the given interval is equally
     * likely to be drawn by `uniform`
     *
     * @param low Lower boundary of the output interval. All values generated will be greater than
     *     or equal to low.
     * @param high Upper boundary of the output interval. All values generated will be less than
     *     high.
     * @param shape the {@link Shape} of the {@link NDArray}
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    NDArray randomUniform(double low, double high, Shape shape, DataType dataType, Device device);

    /**
     * Draw random samples from a normal (Gaussian) distribution. Samples are distributed according
     * to a normal distribution parametrized by mean = 0 and standard deviation = 1 on current
     * device.
     *
     * @param shape Output shape.
     * @return new instance of {@link NDArray}
     */
    default NDArray randomNormal(Shape shape) {
        return randomNormal(0f, 1f, shape, DataType.FLOAT32, getDevice());
    }

    /**
     * Draw random samples from a normal (Gaussian) distribution. Samples are distributed according
     * to a normal distribution parametrized by mean = 0 and standard deviation = 1.
     *
     * @param shape Output shape.
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    default NDArray randomNormal(Shape shape, DataType dataType, Device device) {
        return randomNormal(0d, 1d, shape, dataType, device);
    }

    /**
     * Draw random samples from a normal (Gaussian) distribution. Samples are distributed according
     * to a normal distribution parametrized by {@code *loc*} (mean) and *scale* (standard
     * deviation).
     *
     * @param loc Mean (centre) of the distribution.
     * @param scale Standard deviation (spread or "width") of the distribution.
     * @param shape Output shape.
     * @param dataType the {@link DataType} of the {@link NDArray}
     * @param device the {@link Device} of the {@link NDArray}
     * @return new instance of {@link NDArray}
     */
    NDArray randomNormal(double loc, double scale, Shape shape, DataType dataType, Device device);

    /**
     * Return a single sample from a multinomial distribution. The multinomial distribution is a
     * multivariate generalisation of the binomial distribution. Take an experiment with one of
     * ``p`` possible outcomes. An example of such an experiment is throwing a dice, where the
     * outcome can be 1 through 6. Each sample drawn from the distribution represents n such
     * experiments. Its values, ``X_i = [X_0, X_1, ..., X_p]``, represent the number of times the
     * outcome was ``i``.
     *
     * @param n Number of experiments.
     * @param pValues Probabilities of each of the p different outcomes. These should sum to 1
     *     (however, the last element is always assumed to account for the remaining probability, as
     *     long as ``sum(pvals[:-1]) &lt;= 1)``
     * @return Returns the random NDArray
     */
    NDArray randomMultinomial(int n, NDArray pValues);

    /**
     * Draw samples from a multinomial distribution. The multinomial distribution is a multivariate
     * generalisation of the binomial distribution. Take an experiment with one of ``p`` possible
     * outcomes. An example of such an experiment is throwing a dice, where the outcome can be 1
     * through 6. Each sample drawn from the distribution represents n such experiments. Its values,
     * ``X_i = [X_0, X_1, ..., X_p]``, represent the number of times the outcome was ``i``.
     *
     * @param n Number of experiments.
     * @param pValues Probabilities of each of the p different outcomes. These should sum to 1
     *     (however, the last element is always assumed to account for the remaining probability, as
     *     long as ``sum(pvals[:-1]) &lt;= 1)``
     * @param shape Output shape
     * @return Returns the random NDArray
     */
    NDArray randomMultinomial(int n, NDArray pValues, Shape shape);

    /**
     * Returns parent NDManager.
     *
     * @return parent NDManager
     */
    NDManager getParentManager();

    /**
     * Creates a child NDManager.
     *
     * <p>Child NDManager will inherit default {@link Device} from this NDManager.
     *
     * @return a child NDManager
     */
    NDManager newSubManager();

    /**
     * Creates a child NDManager with specified default {@link Device}.
     *
     * @param device default {@link Device}
     * @return a child NDManager
     */
    NDManager newSubManager(Device device);

    /**
     * Returns default {@link Device} of this NDManager.
     *
     * @return default {@link Device} of this NDManager
     */
    Device getDevice();

    /**
     * Attaches an NDArray or NDManager to this manager.
     *
     * <p>Attached resource will be closed when this manager is closed.
     *
     * @param resourceId unique resourceId
     * @param resource {@link AutoCloseable} resource to be attached
     */
    void attach(String resourceId, AutoCloseable resource);

    /**
     * Detaches an NDArray from this NDManager's lifecycle.
     *
     * <p>The detached NDArray become un-managed, it's user's responsibility to close the resource.
     * Failed to close the resource has to wait on GC to be freed, and might cause out of native
     * memory.
     *
     * @param resourceId resourceId to be remove out of this NDManager's lifecycle
     */
    void detach(String resourceId);

    /**
     * An engine specific generic invocation to native operator.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause portability issue. And user must be aware that native operation may
     * not compatible between each versions.
     *
     * @param operation native operation to performance
     * @param src array of source NDArray
     * @param dest array of destination to save output
     * @param params parameters to be passed to native operator
     * @throws IllegalArgumentException if operation is not supported by Engine
     * @throws EngineException if operation failed in native engine
     */
    void invoke(String operation, NDList src, NDList dest, PairList<String, ?> params);

    /**
     * An engine specific generic invocation to native operator.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause portability issue. And user must be aware that native operation may
     * not compatible between each versions.
     *
     * @param operation native operation to performance
     * @param src array of source NDArray
     * @param params parameters to be passed to native operator
     * @return output array of {@link NDArray}
     * @throws IllegalArgumentException if operation is not supported by Engine
     * @throws EngineException if operation failed in native engine
     */
    NDList invoke(String operation, NDList src, PairList<String, ?> params);

    /** {@inheritDoc} */
    @Override
    void close();
}

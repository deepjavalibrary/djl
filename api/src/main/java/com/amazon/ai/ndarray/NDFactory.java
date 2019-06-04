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
package com.amazon.ai.ndarray;

import com.amazon.ai.Context;
import com.amazon.ai.Translator;
import com.amazon.ai.TranslatorContext;
import com.amazon.ai.engine.Engine;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import com.amazon.ai.ndarray.types.Shape;
import com.amazon.ai.ndarray.types.SparseFormat;

/**
 * NDArray factories are used to create <I>NDArrays</I> (n-dimensional array on native engine).
 *
 * <p>NDFactory is implemented in each deep learning framework {@link Engine}. {@link NDArray}s are
 * resources that allocated in each deep learning framework's native memory space. NDFactory is the
 * key class that manages those native resources.
 *
 * <p>NDArray can only be created through NDFactory, by default, NDArray's lifecycle is attached
 * with creator NDFactory. NDFactory itself implements {@link AutoCloseable}, when NDFactory is
 * closed, all the resource associated with it will be closes as well.
 *
 * <p>A typical place to obtain NDFactory is in {@link Translator#processInput(TranslatorContext,
 * Object)} or {@link Translator#processOutput(TranslatorContext, NDList)}.
 *
 * <p>The following is an example of how to use NDFactory.
 *
 * <pre>
 * public class MyTranslator implements Translator&lt;FloatBuffer, String&gt; {
 *
 *     &#064;Override
 *     public NDList processInput(TranslatorContext ctx, FloatBuffer input) {
 *         <b>NDFactory factory = ctx.getNDFactory();</b>
 *         NDArray array = <b>factory</b>.create(dataDesc);
 *         array.set(input);
 *         return new NDList(array);
 *     } // NDArrays created in this method will be closed after method return.
 * }
 * </pre>
 *
 * <p>NDFactory has a hierarchy structure, it has a single parent NDFactory and can has child
 * NDFactories. When parent NDFactory is closed, all children will be close also.
 *
 * <p>Joule framework manage NDFactory's lifecycle by default. User only need to manage user created
 * child NDFactory. Child NDFactory become useful when user create a large a mount of temporary
 * NDArrays and want to free the resource earlier than parent NDFactory's lifecycle.
 *
 * <p>The following is an example of such use case:
 *
 * <pre>
 * public class MyTranslator implements Translator&lt;List&lt;FloatBuffer&gt;&gt;, String&gt; {
 *
 *     &#064;Override
 *     public NDList processInput(TranslatorContext ctx, List&lt;FloatBuffer&gt; input) {
 *         NDFactory factory = ctx.getNDFactory();
 *         NDArray array = factory.create(dataDesc);
 *         for (int i = 0; i &lt; input.size(); ++i) {
 *             try (<b>NDFactory childFactory = factory.newSubFactory()</b>) {
 *                  NDArray tmp = <b>childFactory</b>.create(itemDataDesc);
 *                  tmp.put(input.get(i);
 *                  array.put(i, tmp);
 *             } // NDArray <i>tmp</i> will be closed here
 *         }
 *         return new NDList(array);
 *     }
 * }
 * </pre>
 *
 * <p>User can also close individual NDArray, NDFactory won't double close them. In certain use
 * case, user might want to return a NDArray to out side of NDFactory's scope,
 *
 * @see NDArray
 * @see Translator
 * @see TranslatorContext#getNDFactory()
 */
public interface NDFactory extends AutoCloseable {

    /**
     * Create an instance of {@link NDArray} with specified {@link Context}, {@link Shape}, {@link
     * DataType} and {@link SparseFormat}.
     *
     * @param context the {@link Context} of the {@link com.amazon.ai.ndarray.NDArray}
     * @param shape the {@link Shape} of the {@link com.amazon.ai.ndarray.NDArray}
     * @param dataType the {@link DataType} of the {@link com.amazon.ai.ndarray.NDArray}
     * @param sparseFormat the {@link SparseFormat} of the {@link com.amazon.ai.ndarray.NDArray}
     * @return new instance of {@link NDArray}
     */
    NDArray create(Context context, Shape shape, DataType dataType, SparseFormat sparseFormat);

    /**
     * Create an instance of {@link NDArray} with specified {@link DataDesc}.
     *
     * @param dataDesc the {@link DataDesc} of the {@link com.amazon.ai.ndarray.NDArray}
     * @return new instance of {@link NDArray}
     */
    NDArray create(DataDesc dataDesc);

    /**
     * Returns parent NDFactory.
     *
     * @return parent NDFactory
     */
    NDFactory getParentFactory();

    /**
     * Returns default {@link Context} or this NDFactory.
     *
     * @return default {@link Context} or this NDFactory
     */
    Context getContext();

    /**
     * Create a child NDFactory.
     *
     * <p>Child NDFactory will inherit default {@link Context} from this NDFactory.
     *
     * @return a child NDFactory
     */
    NDFactory newSubFactory();

    /**
     * Create a child NDFactory with specified default {@link Context}.
     *
     * @param context default {@link Context}
     * @return a child NDFactory
     */
    NDFactory newSubFactory(Context context);

    /**
     * Attach a NDArray or NDFactory to this factory.
     *
     * <p>Attached resource will be closed when this factory is closed.
     *
     * @param resource {@link AutoCloseable} resource to be attached
     */
    void attach(AutoCloseable resource);

    /**
     * Detach a NDArray from this NDFactory's lifecycle.
     *
     * <p>The detached NDArray become un-managed, it's user's responsibility to close the resource.
     * Failed to close the resource has to wait on GC to be freed, and might cause out of native
     * memory.
     *
     * @param resource NDArray to be remove out of this NDFactory's lifecycle
     */
    void detach(AutoCloseable resource);

    /** {@inheritDoc} */
    @Override
    void close();
}

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

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.PairList;

/**
 * {@code ParameterBlock} is an abstract implementation of {@link Block}. It is recommended that all
 * {@link Block} classes that have no children extend the {@code ParameterBlock}.
 */
public abstract class ParameterBlock extends AbstractBlock {

    /** {@inheritDoc} */
    @Override
    public Shape[] initialize(NDManager manager, DataType dataType, Shape... inputShapes) {
        beforeInitialize(inputShapes);
        for (Parameter parameter : getDirectParameters()) {
            parameter.initialize(manager, dataType, inputShapes);
        }
        return getOutputShapes(manager, inputShapes);
    }

    /** {@inheritDoc} */
    @Override
    public final BlockList getChildren() {
        return new BlockList();
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        // FIXME: This is a quick hack for display in jupyter notebook.
        StringBuilder sb = new StringBuilder(200);
        String className = getClass().getSimpleName();
        if (className.endsWith("Block")) {
            className = className.substring(0, className.length() - 5);
        }
        sb.append(className).append('(');
        if (isInitialized()) {
            PairList<String, Shape> inputShapes = describeInput();
            appendShape(sb, inputShapes.values().toArray(new Shape[0]));
            sb.append(" -> ");
            Shape[] outputShapes =
                    getOutputShapes(null, inputShapes.values().toArray(new Shape[0]));
            appendShape(sb, outputShapes);
        } else {
            sb.append("Uninitialized");
        }
        sb.append(')');
        return sb.toString();
    }

    private void appendShape(StringBuilder sb, Shape[] shapes) {
        boolean first = true;
        for (Shape shape : shapes) {
            if (first) {
                first = false;
            } else {
                sb.append(", ");
            }
            long[] sh = shape.getShape();
            int length = sh.length;
            if (length == 0) {
                sb.append("()");
            } else {
                int index = 0;
                if (sh[0] == -1) {
                    --length;
                    index = 1;
                }

                if (length == 0) {
                    sb.append("()");
                } else if (length == 1) {
                    sb.append(sh[index]);
                } else {
                    sb.append('(');
                    for (int i = index; i < sh.length; ++i) {
                        if (i > index) {
                            sb.append(", ");
                        }
                        sb.append(sh[i]);
                    }
                    sb.append(')');
                }
            }
        }
    }
}

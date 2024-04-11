/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.training.listener;

import ai.djl.ndarray.types.Shape;
import ai.djl.util.PairList;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/** One node of the computational graph. */
class Node {

    String name;
    final Node[] src;
    final PairList<String, ?> param;
    boolean isLeaf;
    Shape outputShape;

    public Node(String name, PairList<String, ?> param, Node... src) {
        this.name = name;
        this.param = param;
        this.src = src;
    }

    String toPythonExpression(Map<Node, String> locals, AtomicInteger opCount) {
        return toPythonExpression(locals, opCount, false) + " # " + outputShape;
    }

    String toPythonExpression(Map<Node, String> locals, AtomicInteger opCount, boolean useLocals) {
        if (isLeaf) {
            return name;
        }
        if (useLocals && locals != null && locals.containsKey(this)) {
            return locals.get(this);
        }
        switch (name) {
            case "pick":
                {
                    Object[][] args = {{0}, {1, "indices"}, {"axis", "batch_dims"}};
                    return format("tf.gather", args, locals, opCount);
                }
            case "where":
                {
                    Object[][] args = {{0}, {1, "x"}, {2, "y"}};
                    return format("tf.where", args, locals, opCount);
                }
            case "_npi_slice":
                {
                    Object[][] args = {
                        {0}, {"begin", "begin"}, {"end", "end"}, {"step", "strides"}
                    };
                    return format("tf.strided_slice", args, locals, opCount);
                }
            case "_npi_concatenate":
                {
                    Object[][] args = {{-1}, {"axis", "axis"}};
                    return format("tf.concat", args, locals, opCount);
                }
            case "_np_squeeze":
                {
                    Object[][] args = {{0}, {"axis", "axis"}};
                    return format("tf.squeeze", args, locals, opCount);
                }
            case "_npi_stack":
                {
                    Object[][] args = {{-1}, {"axis", "axis"}};
                    return format("tf.stack", args, locals, opCount);
                }
            case "_npi_split":
                {
                    Object[][] args = {
                        {0}, {"axis", "axis"}, {"num_outputs", "num_or_size_splits"}
                    };
                    return format("tf.split", args, locals, opCount);
                }
            case "_npi_swapaxes":
                {
                    Object[][] args = {{0}, {"dim1", "axis1"}, {"dim2", "axis2"}};
                    return format("tf.experimental.numpy.swapaxes", args, locals, opCount);
                }
            case "_np_repeat":
                {
                    Object[][] args = {{0}, {"repeats", "repeats"}, {"axis", "axis"}};
                    return format("tf.repeat", args, locals, opCount);
                }
            case "_npi_copyto":
                {
                    return src[0].toPythonExpression(locals, opCount, true);
                }
            case "_npi_expand_dims":
                {
                    Object[][] args = {{0}, {"axis", "axis"}};
                    return format("tf.expand_dims", args, locals, opCount);
                }
            case "_npx_log_softmax":
                {
                    Object[][] args = {{0}, {"axis", "axis"}};
                    return format("tf.nn.log_softmax", args, locals, opCount);
                }
            case "_npi_zeros":
                {
                    Object[][] args = {{"shape", "shape"}, {"dtype", "dtype", "tf.dtypes.%s"}};
                    return format("tf.zeros", args, locals, opCount);
                }
            case "_npi_ones":
                {
                    Object[][] args = {{"shape", "shape"}, {"dtype", "dtype", "tf.dtypes.%s"}};
                    return format("tf.ones", args, locals, opCount);
                }
            case "_npi_normal":
                {
                    Object[][] args = {
                        {"size", "shape"},
                        {"loc", "mean"},
                        {"scale", "stddev"},
                        {"dtype", "dtype", "tf.dtypes.%s"}
                    };
                    return format("tf.random.normal", args, locals, opCount);
                }
            case "_npi_uniform":
                {
                    Object[][] args = {
                        {"low", "minval"},
                        {"high", "maxval"},
                        {"shape", "shape"},
                        {"dtype", "dtype", "tf.dtypes.%s"}
                    };
                    return format("tf.random.uniform", args, locals, opCount);
                }
            case "_np_reshape":
                {
                    Object[][] args = {{0}, {"newshape", "shape"}};
                    return format("tf.reshape", args, locals, opCount);
                }
            case "_np_transpose":
                {
                    Object[][] args = {{0}, {"axes", "perm"}};
                    return format("tf.transpose", args, locals, opCount);
                }
            case "_npx_activation":
                {
                    Object[][] args = {{0}};
                    String op =
                            this.param.get("act_type").toString().replace("softrelu", "softplus");
                    return format("tf.nn." + op, args, locals, opCount);
                }
            case "_npx_convolution":
                {
                    String padding = "(0, 0)".equals(this.param.get("pad")) ? "'VALID'" : "'SAME'";
                    Object[][] args = {
                        {0},
                        {1, "filters"},
                        {"stride", "strides"},
                        {"pad", "padding", padding},
                        {"dilate", "dilations"},
                        {null, "data_format", "'NCHW'"}
                    };
                    return addBias(
                            format("tf.nn.convolution", args, locals, opCount),
                            true,
                            locals,
                            opCount);
                }
            case "_npx_pooling":
                {
                    if ("True".equals(this.param.get("global_pool"))) {
                        String op =
                                "avg".equals(this.param.get("pool_type"))
                                        ? "reduce_mean"
                                        : "reduce_max";
                        Object[][] args = {{0}, {null, "axis", "[2, 3]"}};
                        return format("tf." + op, args, locals, opCount);
                    }
                    String padding = "(0, 0)".equals(this.param.get("pad")) ? "'VALID'" : "'SAME'";
                    String poolingType =
                            "avg".equals(this.param.get("pool_type")) ? "'AVG'" : "'MAX'";
                    Object[][] args = {
                        {0},
                        {"kernel", "window_shape"},
                        {"pool_type", "pooling_type", poolingType},
                        {"stride", "strides"},
                        {"pad", "padding", padding},
                        {"dilate", "dilations"},
                        {null, "data_format", "'NCHW'"}
                    };
                    return format("tf.nn.pool", args, locals, opCount);
                }
            case "_npx_batch_norm":
                {
                    Object[][] args = {
                        {0},
                        {1, "scale"},
                        {2, "offset"},
                        {3, "mean"},
                        {4, "variance"},
                        {"eps", "epsilon"},
                        {null, "is_training", "True"},
                        {"momentum", "exponential_avg_factor"},
                        {null, "data_format", "'NCHW'"}
                    };
                    return format("tf.compat.v1.nn.fused_batch_norm", args, locals, opCount);
                }

            case "_npx_embedding":
                {
                    Object[][] args = {
                        {0, "ids"},
                        {1, "params"}
                    };
                    return format("tf.nn.embedding_lookup", args, locals, opCount);
                }
            case "_npx_fully_connected":
                {
                    Object[][] args = {{0}, {1, "b"}, {null, "transpose_b", "True"}};
                    return addBias(
                            format("tf.matmul", args, locals, opCount), false, locals, opCount);
                }
            case "_npi_matmul":
                {
                    Object[][] args = {{0}, {1}};
                    return addBias(
                            format("tf.matmul", args, locals, opCount), false, locals, opCount);
                }
            case "_npi_not_equal_scalar":
                {
                    Object[][] args = {{0}, {"scalar", "y"}};
                    return format("tf.not_equal", args, locals, opCount);
                }
            case "_rdiv_scalar":
                {
                    Object[][] args = {{0}, {"scalar", "y"}};
                    return format("tf.divide", args, locals, opCount);
                }
            case "_npi_add_scalar":
                {
                    Object[][] args = {{0}, {"scalar", "y"}};
                    return format("tf.add", args, locals, opCount);
                }
            case "_npi_add":
                {
                    Object[][] args = {{0}, {1}};
                    return format("tf.add", args, locals, opCount);
                }
            case "_npi_subtract":
                {
                    Object[][] args = {{0}, {1}};
                    return format("tf.subtract", args, locals, opCount);
                }
            case "_npi_mean":
                {
                    Object[][] args = {{0}, {"axis", "axis"}, {"keepdims", "keepdims"}};
                    return format("tf.reduce_mean", args, locals, opCount);
                }
            case "gammaln":
                {
                    Object[][] args = {{0}};
                    return format("tf.gammaln", args, locals, opCount);
                }
            case "_np_sum":
                {
                    Object[][] args = {{0}, {"axis", "axis"}, {"keepdims", "keepdims"}};
                    return format("tf.reduce_sum", args, locals, opCount);
                }
            case "_npi_maximum_scalar":
                {
                    Object[][] args = {{0}, {"scalar", "y"}};
                    return format("tf.maximum", args, locals, opCount);
                }
            case "_npi_multiply_scalar":
                {
                    Object[][] args = {{0}, {"scalar", "y"}};
                    return format("tf.multiply", args, locals, opCount);
                }
            case "_npi_multiply":
                {
                    Object[][] args = {{0}, {1}};
                    return format("tf.multiply", args, locals, opCount);
                }
            case "_npi_true_divide":
                {
                    Object[][] args = {{0}, {1}};
                    return format("tf.divide", args, locals, opCount);
                }
            case "_npi_greater":
                {
                    Object[][] args = {{0}, {1}};
                    return format("tf.greater", args, locals, opCount);
                }
            case "_npi_negative":
                {
                    Object[][] args = {{0}};
                    return format("tf.negative", args, locals, opCount);
                }
            case "_npi_absolute":
                {
                    Object[][] args = {{0}};
                    return format("tf.abs", args, locals, opCount);
                }
            case "_npi_log":
                {
                    Object[][] args = {{0}};
                    return format("tf.log", args, locals, opCount);
                }
            case "_npi_exp":
                {
                    Object[][] args = {{0}};
                    return format("tf.exp", args, locals, opCount);
                }
            default:
                {
                    Stream<Object[]> srcStream =
                            IntStream.range(0, src.length).mapToObj(i -> new Object[] {i});
                    Stream<Object[]> paramStream =
                            param.stream().map(p -> new Object[] {p.getKey(), p.getKey()});
                    Object[][] args =
                            Stream.concat(srcStream, paramStream).toArray(Object[][]::new);
                    return format(name, args, locals, opCount);
                }
        }
    }

    /**
     * Constructs a Python expression for the given operation and formatting arguments.
     *
     * @param op tensorflow operation name
     * @param args array of array of:<br>
     *     [0]: index for {@link #src} or {@link #param} to retrieve argument value, or <code>null
     *     </code><br>
     *     [1]: tensorflow parameter name<br>
     *     [2]: format of argument<br>
     *     [3]: output shape of argument<br>
     * @param locals nodes stored in local Python variables
     * @param opCount operation counter
     * @return the Python expression
     */
    private String format(
            String op, Object[][] args, Map<Node, String> locals, AtomicInteger opCount) {
        StringBuilder sb = new StringBuilder(op + "(\n");
        for (Object[] arg : args) {
            String s = arg.length >= 3 ? String.valueOf(arg[2]) : "%s";
            Shape shape = arg.length >= 4 ? (Shape) arg[3] : null;
            if (Integer.valueOf(-1).equals(arg[0])) {
                s =
                        Stream.of(src)
                                .map(node -> node.toPythonExpression(locals, opCount, true))
                                .map(Node::indent)
                                .collect(Collectors.joining(",\n", "[\n", "\n]"));
            } else if (arg[0] instanceof Integer && src.length > (int) arg[0]) {
                Node node = src[(int) arg[0]];
                s = String.format(s, node.toPythonExpression(locals, opCount, true));
                shape = node.outputShape;
            } else if (this.param.get(String.valueOf(arg[0])) != null) {
                s = String.format(s, this.param.get(String.valueOf(arg[0])));
            } else if (arg[0] != null) {
                continue; // cannot resolve index, so skip
            }
            if (s.startsWith("(") && s.endsWith(")")) {
                s = String.format("[%s]", s.substring(1, s.length() - 1));
            }
            if (arg.length >= 2 && arg[1] != null) {
                s = String.format("%s=%s", arg[1], s);
            }
            sb.append(indent(s) + "," + (shape != null ? " # " + shape : "") + "\n");
        }
        sb.append(
                indent(
                        String.format(
                                "name='%s_%s_',",
                                op.substring(op.lastIndexOf('.') + 1), opCount.incrementAndGet())));
        sb.append("\n)");
        return sb.toString();
    }

    private String addBias(
            String result,
            boolean setChannelFirst,
            Map<Node, String> locals,
            AtomicInteger opCount) {
        if (src.length == 3) {
            Object[][] args = {
                {null, null, result, this.outputShape},
                {2, "bias"},
                {null, "data_format", setChannelFirst ? "'NCHW'" : "None"}
            };
            return format("tf.nn.bias_add", args, locals, opCount);
        }
        return result;
    }

    private void identifyMultipleUsages(Map<Node, Integer> usages) {
        if (isLeaf) {
            return;
        }
        if (usages.compute(this, (key, count) -> count == null ? 1 : count + 1) >= 2) {
            return;
        }
        for (Node node : src) {
            node.identifyMultipleUsages(usages);
        }
        // reposition behind src nodes
        usages.put(this, usages.remove(this));
    }

    String toPythonFunctionBody(AtomicInteger opCount, String result) {
        @SuppressWarnings("PMD.UseConcurrentHashMap")
        Map<Node, Integer> usages = new LinkedHashMap<>();
        identifyMultipleUsages(usages);
        Map<Node, String> locals = new ConcurrentHashMap<>();
        List<String> statements = new ArrayList<>();
        int val = 1;
        int batchnorm = 1;
        for (Map.Entry<Node, Integer> usage : usages.entrySet()) {
            Node node = usage.getKey();
            if (usage.getValue() >= 2) {
                // save the result of an expression that is used multiple times in local variable
                locals.put(node, "val".concat(Integer.toString(val++)));
            } else if ("_npx_batch_norm".equals(node.name)) {
                // local required to assign locals 'running_mean' and 'running_var' at the same time
                locals.put(node, "batchnorm".concat(Integer.toString(batchnorm++)));
            }
        }
        for (Map.Entry<Node, Integer> usage : usages.entrySet()) {
            Node node = usage.getKey();
            if (usage.getValue() >= 2) {
                statements.add(
                        String.format(
                                "%s = %s",
                                locals.get(node), node.toPythonExpression(locals, opCount)));
            } else if ("_npx_batch_norm".equals(node.name)) {
                statements.add(
                        String.format(
                                "(%s, running_mean, running_var) = %s",
                                locals.get(node), node.toPythonExpression(locals, opCount)));
                statements.add(String.format("%s.assign(running_mean)", node.src[3].name));
                statements.add(String.format("%s.assign(running_var)", node.src[4].name));
            }
        }
        statements.add(String.format("%s = %s", result, toPythonExpression(locals, opCount)));
        return statements.stream().map(Node::indent).collect(Collectors.joining("  \n"));
    }

    static String indent(String val) {
        return val.replaceAll("(?m)^", "    ");
    }
}

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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.convolutional.Convolution;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.training.Trainer;
import ai.djl.training.loss.Loss;
import ai.djl.util.Pair;
import ai.djl.util.PairList;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/** Utility class that provides some useful blocks. */
public final class Blocks {

    private Blocks() {}

    /**
     * Inflates the {@link ai.djl.ndarray.NDArray} provided as input to a 2-D {@link
     * ai.djl.ndarray.NDArray} of shape (batch, size).
     *
     * @param array a array to be flattened
     * @return a {@link NDList} that contains the inflated {@link ai.djl.ndarray.NDArray}
     */
    public static NDArray batchFlatten(NDArray array) {
        long batch = array.size(0);
        if (batch == 0) {
            // calculate the size of second dimension manually as using -1 would not work here
            return array.reshape(batch, array.getShape().slice(1).size());
        }
        return array.reshape(batch, -1);
    }

    /**
     * Inflates the {@link ai.djl.ndarray.NDArray} provided as input to a 2-D {@link
     * ai.djl.ndarray.NDArray} of shape (batch, size).
     *
     * @param array a array to be flattened
     * @param size the input size
     * @return a {@link NDList} that contains the inflated {@link ai.djl.ndarray.NDArray}
     * @throws IndexOutOfBoundsException if the input {@link NDList} has more than one {@link
     *     ai.djl.ndarray.NDArray}
     */
    public static NDArray batchFlatten(NDArray array, long size) {
        return array.reshape(-1, size);
    }

    /**
     * Creates a {@link Block} whose forward function applies the {@link #batchFlatten(NDArray)
     * batchFlatten} method.
     *
     * @return a {@link Block} whose forward function applies the {@link #batchFlatten(NDArray)
     *     batchFlatten} method
     */
    public static Block batchFlattenBlock() {
        return LambdaBlock.singleton(Blocks::batchFlatten, "batchFlatten");
    }

    /**
     * Creates a {@link Block} whose forward function applies the {@link #batchFlatten(NDArray)
     * batchFlatten} method. The size of input to the block returned must be batch_size * size.
     *
     * @param size the expected size of each input
     * @return a {@link Block} whose forward function applies the {@link #batchFlatten(NDArray)
     *     batchFlatten} method
     */
    public static Block batchFlattenBlock(long size) {
        return LambdaBlock.singleton(array -> batchFlatten(array, size), "batchFlatten");
    }

    /**
     * Creates a {@link LambdaBlock} that performs the identity function.
     *
     * @return an identity {@link Block}
     */
    public static Block identityBlock() {
        return new LambdaBlock(x -> x, "identity");
    }

    /**
     * Creates a {@link LambdaBlock} that return all-ones NDList.
     *
     * @return an all-ones {@link Block}
     */
    public static Block onesBlock(PairList<DataType, Shape> shapes, String[] names) {
        return new LambdaBlock(
                a -> {
                    Shape[] inShapes = a.getShapes();
                    NDManager manager = a.getManager();
                    NDList list = new NDList(shapes.size());
                    int index = 0;
                    for (Pair<DataType, Shape> pair : shapes) {
                        long[] shape = pair.getValue().getShape().clone();
                        for (int i = 0; i < shape.length; ++i) {
                            if (shape[i] == -1) {
                                shape[i] = inShapes[index].get(i);
                            }
                        }
                        DataType dataType = pair.getKey();
                        NDArray arr = manager.ones(new Shape(shape), dataType);
                        if (names.length == list.size()) {
                            arr.setName(names[index++]);
                        }
                        list.add(arr);
                    }
                    return list;
                },
                "ones");
    }

    /**
     * Returns a string representation of the passed {@link Block} describing the input axes, output
     * axes, and the block's children.
     *
     * @param block the block to describe
     * @param blockName the name to be used for the passed block, or <code>null</code> if its class
     *     name is to be used
     * @param beginAxis skips all axes before this axis; use <code>0</code> to print all axes and
     *     <code>1</code> to skip the batch axis.
     * @return the string representation
     */
    public static String describe(Block block, String blockName, int beginAxis) {
        Shape[] inputShapes = block.isInitialized() ? block.getInputShapes() : null;
        Shape[] outputShapes = inputShapes != null ? block.getOutputShapes(inputShapes) : null;
        StringBuilder sb = new StringBuilder(200);
        if (block instanceof LambdaBlock
                && !LambdaBlock.DEFAULT_NAME.equals(((LambdaBlock) block).getName())) {
            sb.append(((LambdaBlock) block).getName());
        } else if (blockName != null) {
            sb.append(blockName);
        } else {
            sb.append(block.getClass().getSimpleName());
        }
        if (inputShapes != null) {
            sb.append(
                    Stream.of(inputShapes)
                            .map(shape -> shape.slice(beginAxis).toString())
                            .collect(Collectors.joining("+")));
        }
        if (!block.getChildren().isEmpty()) {
            sb.append(" {\n");
            for (Pair<String, Block> pair : block.getChildren()) {
                String child = describe(pair.getValue(), pair.getKey().substring(2), beginAxis);
                sb.append(child.replaceAll("(?m)^", "\t")).append('\n');
            }
            sb.append('}');
        }
        if (outputShapes != null) {
            sb.append(" -> ");
            sb.append(
                    Stream.of(outputShapes)
                            .map(shape -> shape.slice(beginAxis).toString())
                            .collect(Collectors.joining("+")));
        }
        return sb.toString();
    }

    /**
     * Builds an equivalent tensorflow model from the the DJL model using functional or sequential
     * API.
     *
     * @param trainer The trainer containing the DJL model
     * @param functionalApi if <code>true</code>, keras's functional API is used, otherwise the
     *     sequential API. The model should be initialized when using the functional API.
     * @return Python code
     */
    public static String describeAsTensorflow(Trainer trainer, boolean functionalApi) {
        Block block = trainer.getModel().getBlock();
        String model =
                describeAsTensorflow(block, "SequentialBlock", "", functionalApi ? "inputs" : null);
        if (functionalApi) {
            String inputLayer =
                    block.isInitialized()
                            ? String.format(
                                    "inputs = tf.keras.layers.InputLayer(input_shape = %s).output",
                                    block.getInputShapes()[0].slice(1))
                            : "# define input tensor here";
            return String.format(
                    "%s%n%s%nmodel = tf.keras.Model(inputs=inputs, outputs=outputs)%n%nloss = %s",
                    inputLayer, model, describeAsTensorflow(trainer.getLoss()));
        }
        return String.format(
                "model = %s%n%nloss = %s", model, describeAsTensorflow(trainer.getLoss()));
    }

    static String describeAsTensorflow(Loss loss) {
        switch (loss.getClass().getSimpleName()) {
            case "SoftmaxCrossEntropyLoss":
                return "tf.keras.losses.categorical_crossentropy";
            default:
                return "tf.keras.losses.mean_squared_error";
        }
    }

    /**
     * Builds a tensorflow layer equivalent to the passed {@link Block}.
     *
     * @param block the block to translate
     * @param blockName the DJL name of the passed block, or <code>null</code> if the block's class
     *     name is to be used
     * @param pythonName the name to be used for the keras layer name
     * @param input if not <code>null</code>, the input tensor to call the layer with required by
     *     functional API, otherwise sequential API is used
     * @return Python expression for sequential API or Python statements for functional API
     */
    public static String describeAsTensorflow(
            Block block, String blockName, String pythonName, String input) {
        if (block instanceof LambdaBlock
                && !LambdaBlock.DEFAULT_NAME.equals(((LambdaBlock) block).getName())) {
            blockName = ((LambdaBlock) block).getName();
        }
        switch (blockName) {
            case "ParallelBlock":
                {
                    Object[][] args = {{-1}};
                    return format("tf.keras.layers.Concatenate", args, block, pythonName, input);
                }
            case "batchFlatten":
                {
                    Object[][] args = {};
                    return format("tf.keras.layers.Flatten", args, block, pythonName, input);
                }
            case "SequentialBlock":
                {
                    Object[][] args = {{-1}};
                    String op =
                            pythonName.isEmpty()
                                    ? "tf.keras.models.Sequential"
                                    : "tf.keras.Sequential";
                    return format(op, args, block, pythonName, input);
                }
            case "Add":
                {
                    Object[][] args = {{-1}};
                    return format("tf.keras.layers.Add", args, block, pythonName, input);
                }
            case "Linear":
                {
                    Object[][] args = {
                        {block.getOutputShapes(new Shape[] {new Shape(0)})[0].get(0)}
                    };
                    return format("tf.keras.layers.Dense", args, block, pythonName, input);
                }
            case "GELU":
            case "Mish":
            case "ReLU6":
            case "ReLU":
            case "SELU":
            case "sigmoid":
            case "softPlus":
            case "softSign":
            case "Tanh":
                {
                    Object[][] args = {{"tf.keras.activations." + blockName.toLowerCase()}};
                    return format("tf.keras.layers.Activation", args, block, pythonName, input);
                }
            case "identity":
                {
                    Object[][] args = {};
                    return format("tf.keras.layers.Identity", args, block, pythonName, input);
                }
            case "Conv2d":
                {
                    Convolution conv = (Convolution) block;
                    String padding =
                            new Shape(0, 0).equals(conv.getPadding()) ? "'VALID'" : "'SAME'";
                    Object[][] args = {
                        {conv.getFilters(), "filters"},
                        {conv.getKernelShape(), "kernel_size"},
                        {conv.getStride(), "strides"},
                        {null, "padding", padding},
                        {conv.getDilation(), "dilation_rate"},
                        {null, "data_format", "'channels_first'"},
                        {null, "use_bias", conv.isIncludeBias()}
                    };
                    return format("tf.keras.layers.Conv2D", args, block, pythonName, input);
                }

            case "BatchNorm":
                {
                    BatchNorm norm = (BatchNorm) block;
                    Object[][] args = {
                        {norm.getScale(), "scale"},
                        {norm.getCenter(), "center"},
                        {norm.getEpsilon(), "epsilon"},
                        {norm.getMomentum(), "momentum"},
                        {norm.getAxis(), "axis"}
                    };
                    return format(
                            "tf.keras.layers.BatchNormalization", args, block, pythonName, input);
                }

            case "globalAvgPool2d":
                {
                    Object[][] args = {{null, "data_format", "'channels_first'"}};
                    return format(
                            "tf.keras.layers.GlobalAveragePooling2D",
                            args,
                            block,
                            pythonName,
                            input);
                }
            default:
                {
                    Object[][] args = {{-1}};
                    return format(blockName, args, block, pythonName, input);
                }
        }
    }

    static String format(String op, Object[][] args, Block block, String pythonName, String input) {
        String pref = "";
        StringBuilder sb = new StringBuilder(op + "(");
        for (Object[] arg : args) {
            String s = arg.length >= 3 ? String.valueOf(arg[2]) : null;
            if (Integer.valueOf(-1).equals(arg[0])) {
                List<String> nameOfLayers = new ArrayList<>();
                List<String> layers = new ArrayList<>();
                for (Pair<String, Block> pair : block.getChildren()) {
                    String name = pair.getKey().substring(2);
                    String pythonNameOfLayer =
                            pythonName
                                    + (pythonName.isEmpty() ? "" : "_")
                                    + name
                                    + pair.getKey().substring(0, 2);
                    String layer =
                            describeAsTensorflow(pair.getValue(), name, pythonNameOfLayer, input);
                    layers.add(layer);
                    if (input != null) {
                        nameOfLayers.add(
                                layer.substring(
                                        layer.lastIndexOf('\n') + 1, layer.lastIndexOf(" = ")));
                        if (op.endsWith("Sequential")) {
                            input = nameOfLayers.get(nameOfLayers.size() - 1);
                        }
                    }
                }
                if (input != null) {
                    pref = layers.stream().collect(Collectors.joining("\n", "", "\n"));
                    if (!op.endsWith("Sequential")) {
                        input = nameOfLayers.stream().collect(Collectors.joining(", ", "[", "]"));
                    }
                    continue;
                } else {
                    s =
                            layers.stream()
                                    .map(b -> b.replaceAll("(?m)^", "    "))
                                    .collect(Collectors.joining(",\n", "[\n", "\n]"));
                }
            } else if (arg[0] != null) {
                s = arg[0].toString();
            } else if (s == null) {
                continue; // cannot resolve index, so skip
            }
            s = "true".equals(s) ? "True" : "false".equals(s) ? "False" : s;
            if (arg.length >= 2 && arg[1] != null) {
                s = String.format("%s=%s", arg[1], s);
            }
            sb.append(s);
            sb.append(", ");
        }
        String name = pythonName.isEmpty() ? "outputs" : pythonName;
        sb.append(String.format("name='%s'", name));
        sb.append(')');
        if (input != null) {
            if (op.endsWith("Sequential")) {
                return String.format("%s%s = %s", pref, name, input);
            }
            return String.format("%s%s = %s(%s)", pref, name, sb, input);
        }
        return sb.toString();
    }
}

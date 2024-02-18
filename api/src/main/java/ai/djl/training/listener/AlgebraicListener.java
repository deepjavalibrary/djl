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

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.nn.Parameter;
import ai.djl.training.Trainer;
import ai.djl.util.NativeResource;
import ai.djl.util.Pair;
import ai.djl.util.PairList;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/** {@link TrainingListener} that records algebraic operations as Python code. */
public class AlgebraicListener extends TrainingListenerAdapter {

    private static AlgebraicListener currentListener;

    private static final Logger logger = LoggerFactory.getLogger(AlgebraicListener.class);

    private final Map<Object, Node> nodeMap = new ConcurrentHashMap<>();
    private final Map<Object, Node> nodeMapForParameters = new ConcurrentHashMap<>();

    @SuppressWarnings("PMD.UseConcurrentHashMap")
    private final Map<String, Integer> losses = new LinkedHashMap<>();

    @SuppressWarnings("PMD.UseConcurrentHashMap")
    private final Map<String, Integer> predictions = new LinkedHashMap<>();

    private Map<String, String> parameters;
    private String outputFile;
    private AtomicInteger parametersOpCount = new AtomicInteger(0);

    private int numEpoch;

    /**
     * New listener to record algebraic operations into the given file.
     *
     * @param outputFile file to store output - will be overridden if exist
     */
    public AlgebraicListener(String outputFile) {
        this.outputFile = outputFile;
    }

    /** {@inheritDoc} */
    @Override
    public void onEpoch(Trainer trainer) {
        numEpoch++;
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingBatch(Trainer trainer, BatchData batchData) {
        writeParameters(trainer.getModel());
        AtomicInteger opCount = new AtomicInteger(parametersOpCount.get());
        for (Device device : batchData.getLabels().keySet()) {
            NDList data = batchData.getData().get(device);
            NDList preds = batchData.getPredictions().get(device);
            NDList labels = batchData.getLabels().get(device);
            NDArray loss = batchData.getLoss().get(device);
            if (data != null) {
                setLeaf(data, "x");
            }
            if (preds != null) {
                writePredictions(preds, opCount);
            }
            if (preds != null) {
                setLeaf(preds, "prediction");
            }
            if (labels != null) {
                setLeaf(labels, "label");
            }
            if (loss != null) {
                writeLoss(loss, opCount);
            }
        }
        nodeMap.clear();
        nodeMap.putAll(nodeMapForParameters);
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingBegin(Trainer trainer) {
        setCurrentListener(this);
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingEnd(Trainer trainer) {
        try (OutputStream out = Files.newOutputStream(Paths.get(outputFile))) {
            describe(out);
        } catch (IOException e) {
            logger.error("Failed logging algebraic operations", e);
        }
        parameters.clear();
        predictions.clear();
        losses.clear();
        nodeMap.clear();
        nodeMapForParameters.clear();
        setCurrentListener(null);
    }

    private void setLeaf(NDArray x, String name) {
        Node node = get(x);
        if (node == null) {
            return;
        }
        node.name = name;
        node.isLeaf = true;
    }

    private void setLeaf(NDList data, String name) {
        for (NDArray x : data) {
            setLeaf(x, name);
        }
    }

    private void writePredictions(NDList preds, AtomicInteger opCount) {
        String tuple =
                preds.stream()
                        .map(this::getArrayName)
                        .collect(Collectors.joining(", ", "return tf.tuple([", "])"));
        String python =
                preds.stream()
                        .map(pred -> get(pred).toPythonFunctionBody(opCount, getArrayName(pred)))
                        .collect(Collectors.joining("\n", "", "\n" + Node.indent(tuple)));
        predictions.compute(python, (key, count) -> count == null ? 1 : count + 1);
    }

    private String getArrayName(NDArray pred) {
        return pred.getName() != null ? pred.getName() : "result";
    }

    private void writeLoss(NDArray loss, AtomicInteger opCount) {
        String python =
                get(loss).toPythonFunctionBody(opCount, "result")
                        + "\n"
                        + Node.indent("return result");
        losses.compute(python, (key, count) -> count == null ? 1 : count + 1);
    }

    private void describe(OutputStream out) throws IOException {
        PrintStream writer = new PrintStream(out, true, StandardCharsets.US_ASCII.name());
        writer.println("class MyModel(tf.keras.Model):");
        writer.println("  def __init__(self, **kwargs):");
        writer.println("    super().__init__(**kwargs)");
        for (Entry<String, String> param : parameters.entrySet()) {
            writer.println(
                    Node.indent(
                            param.getKey()
                                    + " = tf.Variable(\n"
                                    + Node.indent(param.getValue())
                                    + "\n)"));
        }
        writer.println("");
        for (Entry<String, Integer> pred : predictions.entrySet()) {
            writer.println("## " + pred.getValue());
            writer.println("  def call(self, x):");
            writer.println(pred.getKey());
        }
        writer.println("");
        for (Entry<String, Integer> loss : losses.entrySet()) {
            writer.println("## " + loss.getValue());
            writer.println("def loss(label, prediction):");
            writer.println(loss.getKey());
        }
        writer.println("");
        writer.println(String.format("# number of epochs was %s", numEpoch));
        writer.println(String.format("# number of prediction functions is %s", predictions.size()));
        writer.println(String.format("# number of loss functions is %s", losses.size()));
        writer.println("");
    }

    private void writeParameters(Model model) {
        if (parameters != null) {
            return;
        }
        parameters = new LinkedHashMap<>();
        for (Pair<String, Parameter> pair : model.getBlock().getParameters()) {
            NDArray array = pair.getValue().getArray();
            String initialization =
                    get(array).toPythonExpression(null, parametersOpCount)
                            + (pair.getValue().requiresGradient() ? "" : "\n, trainable = False");
            String pythonClassVariable = "self._" + pair.getKey();
            parameters.put(pythonClassVariable, initialization);
            setLeaf(array, pythonClassVariable);
            nodeMapForParameters.put(key(array), get(array));
        }
    }

    /**
     * Records an algebraic operation that is executed with the given parameters.
     *
     * @param name the name of the operation
     * @param src the input to the operation
     * @param dest the output of the operation
     * @param param parameters for the operation
     */
    public static void record(
            String name, NDArray[] src, NDArray[] dest, PairList<String, ?> param) {
        if (currentListener != null) {
            currentListener.recordInternal(name, src, dest, param);
        }
    }

    private void recordInternal(
            String name, NDArray[] src, NDArray[] dest, PairList<String, ?> param) {
        Node n = new Node(name, param);
        n.src = new ArrayList<>(src.length);
        for (NDArray array : src) {
            Node node = get(array);
            if (node == null) {
                node =
                        new Node(
                                array.getName() != null
                                        ? array.getName()
                                        : "UNKNOWN_ARRAY" + array.getShape(),
                                null);
                nodeMap.put(key(array), n);
                node.outputShape = array.getShape();
            }
            n.src.add(node);
        }
        for (NDArray array : dest) {
            nodeMap.put(key(array), n);
            n.outputShape = array.getShape();
        }
    }

    private Node get(NDArray array) {
        return nodeMap.get(key(array));
    }

    private Object key(NDArray array) {
        return ((NativeResource<?>) array).getHandle();
    }

    private static void setCurrentListener(AlgebraicListener algebraicListener) {
        currentListener = algebraicListener;
    }
}

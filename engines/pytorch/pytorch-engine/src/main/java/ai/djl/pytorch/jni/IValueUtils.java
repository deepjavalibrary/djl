/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.pytorch.jni;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtSymbolBlock;
import ai.djl.util.Pair;
import ai.djl.util.PairList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;

/** IValueUtils is utility class to deal with IValue in PyTorch. */
public final class IValueUtils {

    private static final Pattern PATTERN_LIST = Pattern.compile("\\w+\\[]");
    private static final Pattern PATTERN_TUPLE = Pattern.compile("\\w+\\(\\)");

    private IValueUtils() {}

    /**
     * Runs the forward of PyTorch module.
     *
     * @param block the block that contains PyTorch module
     * @param inputs the input {@link NDList}
     * @param isTrain if running on training mode
     * @return the result {@link NDList}
     */
    public static NDList forward(PtSymbolBlock block, NDList inputs, boolean isTrain) {
        Pair<IValue[], String> inputPair = getInputs(inputs);
        IValue[] ivalues = inputPair.getKey();
        String method = inputPair.getValue();
        long[] iValueHandles = Arrays.stream(ivalues).mapToLong(IValue::getHandle).toArray();
        long result =
                PyTorchLibrary.LIB.moduleRunMethod(
                        block.getHandle(), method, iValueHandles, isTrain);
        PtNDManager manager = (PtNDManager) inputs.get(0).getManager();
        Arrays.stream(ivalues).forEach(IValue::close);
        try (IValue iValue = new IValue(result)) {
            return iValue.toNDList(manager);
        }
    }

    /**
     * Runs the forward of PyTorch module.
     *
     * @param block the block that contains PyTorch module
     * @param inputs the input {@link IValue}
     * @return the result {@link IValue}
     */
    public static IValue forward(PtSymbolBlock block, IValue[] inputs) {
        return runMethod(block, "forward", inputs);
    }

    /**
     * Runs the method of PyTorch module.
     *
     * @param block the block that contains PyTorch module
     * @param methodName the name of method for calling
     * @param inputs the input {@link IValue}
     * @return the result {@link IValue}
     */
    public static IValue runMethod(PtSymbolBlock block, String methodName, IValue... inputs) {
        long[] iValueHandles = Arrays.stream(inputs).mapToLong(IValue::getHandle).toArray();
        return new IValue(
                PyTorchLibrary.LIB.moduleRunMethod(
                        block.getHandle(), methodName, iValueHandles, false));
    }

    private static int addToMap(
            Map<String, Integer> map, String key, List<PairList<String, PtNDArray>> list) {
        return map.computeIfAbsent(
                key,
                k -> {
                    list.add(new PairList<>());
                    return list.size() - 1;
                });
    }

    static Pair<IValue[], String> getInputs(NDList ndList) {
        List<PairList<String, PtNDArray>> outputs = new ArrayList<>();
        Map<String, Integer> indexMap = new ConcurrentHashMap<>();
        String methodName = "forward";
        for (NDArray array : ndList) {
            String name = array.getName();
            if (name != null && name.contains(".")) {
                String[] strings = name.split("\\.", 2);
                int index = addToMap(indexMap, strings[0], outputs);
                PairList<String, PtNDArray> pl = outputs.get(index);
                pl.add(strings[1], (PtNDArray) array);
            } else if (name != null && name.startsWith("module_method:")) {
                methodName = name.substring(14);
            } else if (name != null && PATTERN_LIST.matcher(name).matches()) {
                int index = addToMap(indexMap, name, outputs);
                PairList<String, PtNDArray> pl = outputs.get(index);
                pl.add("[]", (PtNDArray) array);
            } else if (name != null && PATTERN_TUPLE.matcher(name).matches()) {
                int index = addToMap(indexMap, name, outputs);
                PairList<String, PtNDArray> pl = outputs.get(index);
                pl.add("()", (PtNDArray) array);
            } else {
                PairList<String, PtNDArray> pl = new PairList<>();
                pl.add(null, (PtNDArray) array);
                outputs.add(pl);
            }
        }
        IValue[] ret = new IValue[outputs.size()];
        for (int i = 0; i < outputs.size(); ++i) {
            PairList<String, PtNDArray> pl = outputs.get(i);
            String key = pl.get(0).getKey();
            if (key == null) {
                // not List, Dict, Tuple input
                ret[i] = IValue.from(pl.get(0).getValue());
            } else if ("[]".equals(key)) {
                // list
                PtNDArray[] arrays = pl.values().toArray(new PtNDArray[0]);
                ret[i] = IValue.listFrom(arrays);
            } else if ("()".equals(key)) {
                // Tuple
                IValue[] arrays = pl.values().stream().map(IValue::from).toArray(IValue[]::new);
                ret[i] = IValue.tupleFrom(arrays);
            } else {
                Map<String, PtNDArray> map = new ConcurrentHashMap<>();
                for (Pair<String, PtNDArray> pair : pl) {
                    map.put(pair.getKey(), pair.getValue());
                }
                ret[i] = IValue.stringMapFrom(map);
            }
        }
        return new Pair<>(ret, methodName);
    }

    /**
     * Converts ndList to IValue.
     *
     * @param ndList the NDList to convert
     * @param dims the shape of the output
     * @return the result {@link IValue}
     */
    public static IValue toTupleIValue(NDList ndList, long[] dims) {
        return toTupleIValueRecur(ndList, dims, 0, 0).getKey();
    }

    /**
     * Helper function.
     *
     * @param ndList the NDList to convert
     * @param dims the shape of the output
     * @param startCount the start index of the current recursion level
     * @param level the recursion level
     * @return the result
     */
    private static Pair<IValue, Integer> toTupleIValueRecur(
            NDList ndList, long[] dims, int startCount, int level) {
        if (startCount > ndList.size()) {
            throw new IllegalArgumentException("startCount illegal");
        }
        if (dims.length - 1 == level) {
            long dim = dims[level];
            List<PtNDArray> vector = new ArrayList<>();
            for (int i = startCount; i < startCount + dim; i++) {
                vector.add((PtNDArray) ndList.get(i));
            }
            IValue[] output = vector.stream().map(IValue::from).toArray(IValue[]::new);
            return new Pair<>(IValue.tupleFrom(output), Math.toIntExact((startCount + dim)));
        }

        IValue[] output = new IValue[Math.toIntExact(dims[0])];
        for (int j = 0; j < dims[level]; j++) {
            Pair<IValue, Integer> p = toTupleIValueRecur(ndList, dims, startCount, level + 1);
            startCount = p.getValue();
            output[j] = p.getKey();
        }
        return new Pair<>(IValue.tupleFrom(output), startCount);
    }
}

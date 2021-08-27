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
        IValue[] iValues = getInputs(inputs);
        long[] iValueHandles = Arrays.stream(iValues).mapToLong(IValue::getHandle).toArray();
        long result = PyTorchLibrary.LIB.moduleForward(block.getHandle(), iValueHandles, isTrain);
        PtNDManager manager = (PtNDManager) inputs.get(0).getManager();
        Arrays.stream(iValues).forEach(IValue::close);
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
    public static IValue forward(PtSymbolBlock block, IValue... inputs) {
        long[] handles = Arrays.stream(inputs).mapToLong(IValue::getHandle).toArray();
        return new IValue(PyTorchLibrary.LIB.moduleForward(block.getHandle(), handles, false));
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

    private static IValue[] getInputs(NDList ndList) {
        List<PairList<String, PtNDArray>> outputs = new ArrayList<>();
        Map<String, Integer> indexMap = new ConcurrentHashMap<>();
        for (NDArray array : ndList) {
            String name = array.getName();
            if (name != null && name.contains(".")) {
                String[] strings = name.split("\\.", 2);
                int index = addToMap(indexMap, strings[0], outputs);
                PairList<String, PtNDArray> pl = outputs.get(index);
                pl.add(strings[1], (PtNDArray) array);
            } else if (name != null && Pattern.matches("\\w+\\[]", name)) {
                int index = addToMap(indexMap, name, outputs);
                PairList<String, PtNDArray> pl = outputs.get(index);
                pl.add("[]", (PtNDArray) array);
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
                // not List, Dict input
                ret[i] = IValue.from(pl.get(0).getValue());
            } else if ("[]".equals(key)) {
                // list
                PtNDArray[] arrays = pl.values().toArray(new PtNDArray[0]);
                ret[i] = IValue.listFrom(arrays);
            } else {
                Map<String, PtNDArray> map = new ConcurrentHashMap<>();
                for (Pair<String, PtNDArray> pair : pl) {
                    map.put(pair.getKey(), pair.getValue());
                }
                ret[i] = IValue.stringMapFrom(map);
            }
        }
        return ret;
    }
}

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
package org.apache.mxnet.engine;

import com.amazon.ai.Model;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import com.amazon.ai.ndarray.types.Shape;
import com.amazon.ai.util.Pair;
import com.amazon.ai.util.PairList;
import com.amazon.ai.util.Utils;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MxModel implements Model, AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(MxModel.class);

    private Symbol symbol;
    private PairList<String, MxNDArray> parameters;
    private String[] synset;
    private String[] optimizerStates;
    private String[] fixedParameters;
    private DataDesc[] inputData;

    MxModel(
            Symbol symbol,
            PairList<String, MxNDArray> parameters,
            String[] synset,
            String[] optimizerStates) {
        this.symbol = symbol;
        this.parameters = parameters;
        this.synset = synset;
        this.optimizerStates = optimizerStates;
    }

    static MxModel loadModel(String prefix, int epoch) throws IOException {
        return loadModel(MxNDFactory.SYSTEM_FACTORY, prefix, epoch);
    }

    static MxModel loadModel(MxNDFactory factory, String prefix, int epoch) throws IOException {
        Symbol symbol = Symbol.load(factory, prefix + "-symbol.json");
        String paramFile = String.format("%s-%04d.params", prefix, epoch);
        String stateFile = String.format("%s-%04d.states", prefix, epoch);
        File synsetFile = new File(new File(paramFile).getParentFile(), "synset.txt");

        PointerByReference namesRef = new PointerByReference();
        Pointer[] handles = JnaUtils.loadNdArray(paramFile, namesRef);
        String[] names = namesRef.getValue().getStringArray(0, handles.length);

        PairList<String, MxNDArray> parameters = new PairList<>();

        for (int i = 0; i < names.length; ++i) {
            String[] pair = names[i].split(":", 2);
            MxNDArray array = factory.create(handles[i]);
            parameters.add(pair[1], array);
        }

        String[] synset = loadSynset(synsetFile);
        String[] stateNames = JnaUtils.readLines(new File(stateFile)).toArray(JnaUtils.EMPTY_ARRAY);

        JnaUtils.waitAll();

        return new MxModel(symbol, parameters, synset, stateNames);
    }

    public Symbol getSymbol() {
        return symbol;
    }

    public PairList<String, MxNDArray> getParameters() {
        return parameters;
    }

    /** {@inheritDoc} */
    @Override
    public Model cast(DataType dataType) {
        if (parameters.get(0).getValue().getDataType() == dataType) {
            logger.info("You are casting the model to its original type!");
            return this;
        }
        PairList<String, MxNDArray> newParam = new PairList<>();
        for (Pair<String, MxNDArray> pair : parameters) {
            newParam.add(pair.getKey(), pair.getValue().asType(dataType, true));
        }
        return new MxModel(symbol, newParam, synset, optimizerStates);
    }

    public String[] getOptimizerStates() {
        return optimizerStates;
    }

    public void setOptimizerStates(String[] optimizerStates) {
        validate(optimizerStates, "state", true);
        this.optimizerStates = optimizerStates;
    }

    public String[] getFixedParameters() {
        return fixedParameters;
    }

    public void setFixedParameters(String[] fixedParameters) {
        validate(fixedParameters, "fixed_param", true);
        this.fixedParameters = fixedParameters;
    }

    public void saveCheckpoint(
            String prefix,
            int epoch,
            Symbol symbol,
            Map<String, MxNDArray> argParams,
            Map<String, MxNDArray> auxParams) {
        symbol.save(prefix + "-symbol.json");
        String paramName = String.format("%s-%04d.params", prefix, epoch);

        Pointer[] pointers = new Pointer[argParams.size() + auxParams.size()];
        String[] keys = new String[pointers.length];
        int i = 0;
        for (Map.Entry<String, MxNDArray> entry : argParams.entrySet()) {
            keys[i] = "arg:" + entry.getKey();
            pointers[i] = entry.getValue().getHandle();
            ++i;
        }
        for (Map.Entry<String, MxNDArray> entry : auxParams.entrySet()) {
            keys[i] = "aux:" + entry.getKey();
            pointers[i] = entry.getValue().getHandle();
            ++i;
        }

        JnaUtils.saveNdArray(paramName, pointers, keys);
    }

    public void save(File dir, String name, int epoch) {}

    /** {@inheritDoc} */
    @Override
    public void close() {
        symbol.close();

        for (int i = 0; i < parameters.size(); ++i) {
            MxNDArray array = parameters.valueAt(i);
            array.close();
        }
    }

    public static String[] loadSynset(File synsetFile) {
        if (synsetFile.exists()) {
            try {
                List<String> output = Files.readAllLines(synsetFile.toPath());
                ListIterator<String> it = output.listIterator();
                while (it.hasNext()) {
                    String synsetLemma = it.next();
                    it.set(synsetLemma.substring(synsetLemma.indexOf(' ') + 1));
                }
                return output.toArray(JnaUtils.EMPTY_ARRAY);
            } catch (IOException e) {
                logger.warn("Error opening synset file " + synsetFile, e);
            }
        }
        return JnaUtils.EMPTY_ARRAY;
    }

    /** {@inheritDoc} */
    @Override
    public String[] getSynset() {
        return synset;
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc[] describeInput() {
        if (inputData == null) {
            String[] allNames = symbol.getAllNames();
            Map<String, Integer> map = new ConcurrentHashMap<>(allNames.length * 3 / 2);
            int index = 0;
            for (String name : allNames) {
                map.put(name, index++);
            }
            for (String name : parameters.keys()) {
                map.remove(name);
            }
            inputData = new DataDesc[map.size()];

            index = 0;
            for (String name : map.keySet()) {
                inputData[index++] = new DataDesc(new Shape(), name);
            }
        }
        return inputData;
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc[] describeOutput() {
        return null;
    }

    private void validate(String[] names, String typeName, boolean required) {
        if (names == null || names.length == 0) {
            return;
        }

        String[] args = symbol.getArgParams();
        for (String name : names) {
            if (!Utils.contains(args, name)) {
                String msg =
                        String.format(
                                "Input %s_%s is not found in symbol.list_arguments().",
                                typeName, name);
                if (required) {
                    throw new IllegalArgumentException(msg);
                }
                logger.warn(msg);
            }
        }
    }
}

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
import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MxModel implements Model, AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(MxModel.class);

    private Symbol symbol;
    private PairList<String, MxNDArray> argParams;
    private PairList<String, MxNDArray> auxParams;
    private String[] synset;
    private String[] labelNames;
    private String[] optimizerStates;
    private String[] fixedParameters;
    private DataDesc[] inputData;

    MxModel(
            Symbol symbol,
            PairList<String, MxNDArray> argParams,
            PairList<String, MxNDArray> auxParams,
            String[] synset,
            String[] optimizerStates) {
        this.symbol = symbol;
        this.argParams = argParams;
        this.auxParams = auxParams;
        this.synset = synset;
        this.optimizerStates = optimizerStates;
        labelNames = new String[] {"softmax_label"};
    }

    public static MxModel loadModel(String prefix, int epoch) throws IOException {
        return loadModel(MxNDFactory.SYSTEM_FACTORY, prefix, epoch);
    }

    public static MxModel loadModel(MxNDFactory factory, String prefix, int epoch)
            throws IOException {
        Symbol symbol = Symbol.load(factory, prefix + "-symbol.json");
        String paramFile = String.format("%s-%04d.params", prefix, epoch);
        String stateFile = String.format("%s-%04d.states", prefix, epoch);
        File synsetFile = new File(new File(paramFile).getParentFile(), "synset.txt");

        PointerByReference namesRef = new PointerByReference();
        Pointer[] handles = JnaUtils.loadNdArray(paramFile, namesRef);
        String[] names = namesRef.getValue().getStringArray(0, handles.length);

        PairList<String, MxNDArray> argParams = new PairList<>();
        PairList<String, MxNDArray> auxParams = new PairList<>();
        for (int i = 0; i < names.length; ++i) {
            String[] pair = names[i].split(":", 2);
            MxNDArray array = factory.create(handles[i]);
            if ("arg".equals(pair[0])) {
                argParams.add(pair[1], array);
            } else if ("aux".equals(pair[0])) {
                auxParams.add(pair[1], array);
            } else {
                throw new IllegalStateException("Unknown parameter: " + pair[0]);
            }
        }

        String[] synset = loadSynset(synsetFile);
        String[] stateNames = JnaUtils.readLines(new File(stateFile)).toArray(JnaUtils.EMPTY_ARRAY);

        JnaUtils.waitAll();

        return new MxModel(symbol, argParams, auxParams, synset, stateNames);
    }

    public Symbol getSymbol() {
        return symbol;
    }

    public PairList<String, MxNDArray> getArgParams() {
        return argParams;
    }

    public PairList<String, MxNDArray> getAuxParams() {
        return auxParams;
    }

    public void setDataNames(DataDesc... inputData) {
        this.inputData = inputData;
    }

    @Override
    public Model cast(DataType dataType) {
        return null;
    }

    public String[] getLabelNames() {
        if (labelNames == null) {
            return JnaUtils.EMPTY_ARRAY;
        }
        return labelNames;
    }

    public void setLabelNames(String... labelNames) {
        validate(labelNames, "label", false);
        this.labelNames = labelNames;
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

    @Override
    public void close() {
        symbol.close();

        for (int i = 0; i < argParams.size(); ++i) {
            MxNDArray array = argParams.valueAt(i);
            array.close();
        }
        for (int i = 0; i < auxParams.size(); ++i) {
            MxNDArray array = auxParams.valueAt(i);
            array.close();
        }
    }

    public static String[] loadSynset(File synsetFile) {
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
        return JnaUtils.EMPTY_ARRAY;
    }

    @Override
    public String[] getSynset() {
        return synset;
    }

    @Override
    public DataDesc[] describeInput() {
        return inputData;
    }

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

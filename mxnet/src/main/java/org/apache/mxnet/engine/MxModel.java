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

import com.amazon.ai.Block;
import com.amazon.ai.Context;
import com.amazon.ai.Model;
import com.amazon.ai.ndarray.types.DataType;
import com.amazon.ai.ndarray.types.Shape;
import com.amazon.ai.util.PairList;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MxModel implements Model, AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(MxModel.class);

    private static final String[] EMPTY = new String[0];

    private Symbol symbol;
    private PairList<String, MxNDArray> argParams;
    private PairList<String, MxNDArray> auxParams;
    private String[] synset;
    private String[] labelNames;
    private String[] dataNames;
    private String[] optimizerStates;

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
        return loadModel(null, prefix, epoch);
    }

    public static MxModel loadModel(ResourceAllocator alloc, String prefix, int epoch)
            throws IOException {
        Symbol symbol = Symbol.load(alloc, prefix + "-symbol.json");
        String paramFile = String.format("%s-%04d.params", prefix, epoch);
        String stateFile = String.format("%s-%04d.states", prefix, epoch);
        File synsetFile = new File(new File(paramFile).getParentFile(), "synset.txt");

        PointerByReference namesRef = new PointerByReference();
        Pointer[] handles = JnaUtils.loadNdArray(paramFile, namesRef);
        String[] names = namesRef.getValue().getStringArray(0, handles.length);

        Context context = Context.cpu();

        List<String> argParamNames = new ArrayList<>();
        List<String> auxParamNames = new ArrayList<>();
        List<MxNDArray> argParamData = new ArrayList<>();
        List<MxNDArray> auxParamData = new ArrayList<>();
        for (int i = 0; i < names.length; ++i) {
            String[] pair = names[i].split(":", 2);
            if ("arg".equals(pair[0])) {
                argParamNames.add(pair[1]);
                argParamData.add(
                        new MxNDArray(alloc, context, null, null, DataType.FLOAT32, handles[i]));
            } else if ("aux".equals(pair[0])) {
                auxParamNames.add(pair[1]);
                auxParamData.add(
                        new MxNDArray(alloc, context, null, null, DataType.FLOAT32, handles[i]));
            } else {
                throw new IllegalStateException("Unknown parameter: " + pair[0]);
            }
        }
        PairList<String, MxNDArray> argParams = new PairList<>(argParamNames, argParamData);
        PairList<String, MxNDArray> auxParams = new PairList<>(auxParamNames, auxParamData);

        String[] synset = loadSynset(synsetFile);
        String[] stateNames = JnaUtils.readLines(new File(stateFile)).toArray(new String[0]);

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

    public String[] getSynset() {
        if (synset == null) {
            return EMPTY;
        }
        return synset;
    }

    @Override
    public String[] getDataNames() {
        return dataNames;
    }

    @Override
    public void setDataNames(String... dataNames) {
        this.dataNames = dataNames;
    }

    public String[] getLabelNames() {
        if (labelNames == null) {
            return EMPTY;
        }
        return labelNames;
    }

    public void setLabelNames(String... labelNames) {
        this.labelNames = labelNames;
    }

    public String[] getOptimizerStates() {
        return optimizerStates;
    }

    public void setOptimizerStates(String[] optimizerStates) {
        this.optimizerStates = optimizerStates;
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
        for (MxNDArray nd : argParams.values()) {
            nd.close();
        }
        for (MxNDArray nd : auxParams.values()) {
            nd.close();
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
            return output.toArray(new String[0]); // NOPMD
        } catch (IOException e) {
            logger.warn("Error opening synset file " + synsetFile, e);
        }
        return null;
    }

    @Override
    public Block getNetwork() {
        return null;
    }

    @Override
    public String[] getLabels() {
        return labelNames;
    }

    @Override
    public Shape getInputShape() {
        return symbol.getInputShape();
    }

    @Override
    public Shape getOutputShape() {
        return symbol.getOutputShape();
    }
}

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
package org.apache.mxnet.model;

import com.amazon.ai.util.PairList;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import java.io.File;
import java.io.IOException;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import org.apache.mxnet.Context;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.jna.MxnetLibrary;
import org.apache.mxnet.types.DataType;

public class MxModel implements AutoCloseable {

    private static final String[] EMPTY = new String[0];

    private Symbol symbol;
    private PairList<String, NdArray> argParams;
    private PairList<String, NdArray> auxParams;
    private String[] synset;
    private String[] labelNames;
    private String[] optimizerStates;

    MxModel(
            Symbol symbol,
            PairList<String, NdArray> argParams,
            PairList<String, NdArray> auxParams,
            String[] synset,
            String[] optimizerStates) {
        this.symbol = symbol;
        this.argParams = argParams;
        this.auxParams = auxParams;
        this.synset = synset;
        this.optimizerStates = optimizerStates;
        labelNames = new String[] {"softmax_label"};
    }

    public static MxModel loadSavedModel(String prefix, int epoch) throws IOException {
        return loadSavedModel(null, prefix, epoch);
    }

    public static MxModel loadSavedModel(ResourceAllocator alloc, String prefix, int epoch)
            throws IOException {
        Symbol symbol = Symbol.load(alloc, prefix + "-symbol.json");
        String paramFile = String.format("%s-%04d.params", prefix, epoch);
        String stateFile = String.format("%s-%04d.states", prefix, epoch);
        File synsetFile = new File(new File(paramFile).getParentFile(), "synset.txt");

        IntBuffer outSize = IntBuffer.wrap(new int[1]);
        IntBuffer nameSize = IntBuffer.wrap(new int[1]);
        PointerByReference ref = new PointerByReference();
        PointerByReference nameRef = new PointerByReference();
        JnaUtils.checkCall(
                MxnetLibrary.INSTANCE.MXNDArrayLoad(paramFile, outSize, ref, nameSize, nameRef));

        int ndArrayCount = outSize.get();
        int nameCount = nameSize.get();
        if (ndArrayCount != nameCount) {
            throw new IllegalStateException(
                    "Mismatch between names and arrays in checkpoint file: " + paramFile);
        }

        String[] names = nameRef.getValue().getStringArray(0, nameCount);
        Pointer[] handles = ref.getValue().getPointerArray(0, nameCount);

        Context context = Context.cpu();

        List<String> argParamNames = new ArrayList<>();
        List<String> auxParamNames = new ArrayList<>();
        List<NdArray> argParamData = new ArrayList<>();
        List<NdArray> auxParamData = new ArrayList<>();
        for (int i = 0; i < nameCount; ++i) {
            String[] pair = names[i].split(":", 2);
            if ("arg".equals(pair[0])) {
                argParamNames.add(pair[1]);
                argParamData.add(
                        new NdArray(alloc, context, null, null, DataType.FLOAT32, handles[i]));
            } else if ("aux".equals(pair[0])) {
                auxParamNames.add(pair[1]);
                auxParamData.add(
                        new NdArray(alloc, context, null, null, DataType.FLOAT32, handles[i]));
            } else {
                throw new IllegalStateException("Unknown parameter: " + pair[0]);
            }
        }
        PairList<String, NdArray> argParams = new PairList<>(argParamNames, argParamData);
        PairList<String, NdArray> auxParams = new PairList<>(auxParamNames, auxParamData);

        String[] synset = loadSynset(synsetFile);
        String[] stateNames = JnaUtils.readLines(new File(stateFile)).toArray(new String[0]);

        JnaUtils.waitAll();

        return new MxModel(symbol, argParams, auxParams, synset, stateNames);
    }

    public Symbol getSymbol() {
        return symbol;
    }

    public PairList<String, NdArray> getArgParams() {
        return argParams;
    }

    public PairList<String, NdArray> getAuxParams() {
        return auxParams;
    }

    public String[] getSynset() {
        if (synset == null) {
            return EMPTY;
        }
        return synset;
    }

    public String[] getLabelNames() {
        if (labelNames == null) {
            return EMPTY;
        }
        return labelNames;
    }

    public void setLabelNames(String[] labelNames) {
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
            Map<String, NdArray> argParams,
            Map<String, NdArray> auxParams) {
        symbol.save(prefix + "-symbol.json");
        String paramName = String.format("%s-%04d.params", prefix, epoch);

        Pointer[] pointers = new Pointer[argParams.size() + auxParams.size()];
        String[] keys = new String[pointers.length];
        int i = 0;
        for (Map.Entry<String, NdArray> entry : argParams.entrySet()) {
            keys[i] = "arg:" + entry.getKey();
            pointers[i] = entry.getValue().getHandle();
            ++i;
        }
        for (Map.Entry<String, NdArray> entry : auxParams.entrySet()) {
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
        for (NdArray nd : argParams.values()) {
            nd.close();
        }
        for (NdArray nd : auxParams.values()) {
            nd.close();
        }
    }

    public static String[] loadSynset(File synsetFile) {
        try {
            List<String> output = Files.readAllLines(synsetFile.toPath());
            ListIterator<String> it = output.listIterator();
            while (it.hasNext()) {
                String synsetLemma = it.next();
                it.set(synsetLemma.substring(synsetLemma.indexOf(" ") + 1));
            }
            return output.toArray(new String[0]);
        } catch (IOException e) {
            System.err.println("Error opening synset file " + synsetFile + ": " + e);
        }
        return null;
    }
}

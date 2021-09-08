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
package ai.djl.fasttext.jni;

import ai.djl.modality.Classifications;
import ai.djl.util.NativeResource;
import java.util.ArrayList;
import java.util.List;

/** A class containing utilities to interact with the fastText JNI layer. */
@SuppressWarnings("MissingJavadocMethod")
public final class FtWrapper extends NativeResource<Long> {

    private static RuntimeException libraryStatus;

    static {
        try {
            LibUtils.loadLibrary();
        } catch (RuntimeException e) {
            libraryStatus = e;
        }
    }

    private FtWrapper() {
        super(FastTextLibrary.LIB.createFastText());
    }

    public static FtWrapper newInstance() {
        if (libraryStatus != null) {
            throw libraryStatus;
        }
        return new FtWrapper();
    }

    public void loadModel(String modelFilePath) {
        FastTextLibrary.LIB.loadModel(getHandle(), modelFilePath);
    }

    public boolean checkModel(String modelFilePath) {
        return FastTextLibrary.LIB.checkModel(modelFilePath);
    }

    public void unloadModel() {
        FastTextLibrary.LIB.unloadModel(getHandle());
    }

    public String getModelType() {
        return FastTextLibrary.LIB.getModelType(getHandle());
    }

    public Classifications predictProba(String text, int topK, String labelPrefix) {
        String[] labels = new String[topK];
        float[] probs = new float[topK];

        int size = FastTextLibrary.LIB.predictProba(getHandle(), text, topK, labels, probs);

        List<String> classes = new ArrayList<>(size);
        List<Double> probabilities = new ArrayList<>(size);
        for (int i = 0; i < size; ++i) {
            String label = labels[i];
            if (label.startsWith(labelPrefix)) {
                label = label.substring(labelPrefix.length());
            }
            classes.add(label);
            probabilities.add((double) probs[i]);
        }
        return new Classifications(classes, probabilities);
    }

    public float[] getDataVector(String word) {
        return FastTextLibrary.LIB.getWordVector(getHandle(), word);
    }

    public void runCmd(String[] args) {
        FastTextLibrary.LIB.runCmd(args);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        Long pointer = handle.getAndSet(null);
        if (pointer != null) {
            FastTextLibrary.LIB.freeFastText(pointer);
        }
    }
}

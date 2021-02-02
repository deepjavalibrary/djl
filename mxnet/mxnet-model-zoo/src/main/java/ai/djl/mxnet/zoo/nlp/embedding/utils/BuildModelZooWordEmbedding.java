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
package ai.djl.mxnet.zoo.nlp.embedding.utils;

import ai.djl.Model;
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding;
import ai.djl.ndarray.NDArray;
import ai.djl.util.Utils;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
 * A utility to build embeddings into DJL format from Gluon-NLP format.
 *
 * <p>This utility should be called after running convertEmbeddings.py
 */
public final class BuildModelZooWordEmbedding {

    private BuildModelZooWordEmbedding() {}

    /**
     * Runs main build embeddings after editing.
     *
     * @param args the arguments
     * @throws IOException thrown if unable to read files in directory
     */
    public static void main(String[] args) throws IOException {
        // EDIT THESE STRINGS TO THE EMBEDDING DIR AND NAME
        buildEmbedding("", "");
    }

    private static void buildEmbedding(String dir, String name) throws IOException {
        Path path = Paths.get(dir);
        Model model = Model.newInstance(name);
        NDArray idxToVec =
                model.getNDManager().load(path.resolve("idx_to_vec.mx")).singletonOrThrow();
        List<String> idxToToken = Utils.readLines(path.resolve("idx_to_token.txt"));
        TrainableWordEmbedding embedding = new TrainableWordEmbedding(idxToVec, idxToToken);
        model.setBlock(embedding);
        model.save(path, name);
    }
}

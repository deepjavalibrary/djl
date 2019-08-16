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
package org.apache.mxnet.zoo.cv.image_classification;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.mxnet.zoo.ModelNotFoundException;
import org.apache.mxnet.zoo.ModelZoo;
import org.apache.mxnet.zoo.ZooModel;
import software.amazon.ai.Model;
import software.amazon.ai.TranslatorContext;
import software.amazon.ai.modality.Classification;
import software.amazon.ai.modality.cv.ImageTranslator;
import software.amazon.ai.modality.cv.Images;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.repository.Artifact;
import software.amazon.ai.repository.MRL;
import software.amazon.ai.repository.Metadata;
import software.amazon.ai.repository.Repository;
import software.amazon.ai.repository.VersionRange;
import software.amazon.ai.util.Utils;

public class Resnet {

    private static final String ARTIFACT_ID = "resnet";
    private static final MRL RESNET =
            new MRL(MRL.Model.CV.IMAGE_CLASSIFICATION, ModelZoo.GROUP_ID, ARTIFACT_ID);

    private Repository repository;
    private Metadata metadata;

    public Resnet(Repository repository) {
        this.repository = repository;
    }

    private void locateMetadata() throws IOException, ModelNotFoundException {
        if (metadata == null) {
            metadata = repository.locate(RESNET);
            if (metadata == null) {
                throw new ModelNotFoundException("Resnet models not found.");
            }
        }
    }

    public ZooModel<BufferedImage, List<Classification>> loadModel(Map<String, String> criteria)
            throws IOException, ModelNotFoundException {
        locateMetadata();
        Artifact artifact = match(criteria);
        if (artifact == null) {
            throw new ModelNotFoundException("Model not found.");
        }
        repository.prepare(artifact);
        Path dir = repository.getCacheDirectory();
        String relativePath = artifact.getResourceUri().getPath();
        Path modelPath = dir.resolve(relativePath);
        Model model = Model.loadModel(modelPath, artifact.getName());
        return new ZooModel<>(
                model,
                new ImageTranslator<List<Classification>>() {

                    private int topK = 5;
                    private int imageWidth = 224;
                    private int imageHeight = 224;
                    private DataDesc dataDesc =
                            new DataDesc(new Shape(1, 3, imageWidth, imageHeight), "data");

                    @Override
                    public NDList processInput(TranslatorContext ctx, BufferedImage input) {
                        BufferedImage image = Images.centerCrop(input);
                        image = Images.resizeImage(image, imageWidth, imageHeight);
                        FloatBuffer buffer = Images.toFloatBuffer(image);
                        NDArray array = ctx.getNDManager().create(buffer, dataDesc);
                        array.divi(255);

                        return new NDList(array);
                    }

                    @Override
                    public List<Classification> processOutput(TranslatorContext ctx, NDList list)
                            throws IOException {
                        Model model = ctx.getModel();
                        NDArray array = list.get(0).get(0);

                        long length = array.getShape().head();
                        length = Math.min(length, topK);
                        List<Classification> ret = new ArrayList<>(Math.toIntExact(length));
                        NDArray sorted = array.argsort(-1, false);
                        NDArray top = sorted.get(":" + topK);

                        float[] probabilities = array.softmax(-1).toFloatArray();
                        int[] indices = top.toIntArray();

                        List<String> synset = model.getArtifact("synset.txt", Utils::readLines);
                        for (int i = 0; i < topK; ++i) {
                            int index = indices[i];
                            String className = synset.get(index);
                            Classification out =
                                    new Classification(className, probabilities[index]);
                            ret.add(out);
                        }
                        return ret;
                    }
                });
    }

    public Artifact match(Map<String, String> criteria) {
        List<Artifact> list = search(criteria);
        if (list.isEmpty()) {
            return null;
        }

        list.sort(Artifact.COMPARATOR);
        return list.get(0);
    }

    public List<Artifact> search(Map<String, String> criteria) {
        return metadata.search(VersionRange.parse("0.0.1"), criteria);
    }
}

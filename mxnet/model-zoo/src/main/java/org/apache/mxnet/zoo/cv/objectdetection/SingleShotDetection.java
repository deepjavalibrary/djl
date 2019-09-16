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
package org.apache.mxnet.zoo.cv.objectdetection;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.mxnet.zoo.ModelNotFoundException;
import org.apache.mxnet.zoo.ModelZoo;
import org.apache.mxnet.zoo.ZooModel;
import software.amazon.ai.Model;
import software.amazon.ai.modality.cv.DetectedObject;
import software.amazon.ai.modality.cv.ImageTranslator;
import software.amazon.ai.modality.cv.Images;
import software.amazon.ai.modality.cv.Rectangle;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.repository.Artifact;
import software.amazon.ai.repository.MRL;
import software.amazon.ai.repository.Metadata;
import software.amazon.ai.repository.Repository;
import software.amazon.ai.repository.VersionRange;
import software.amazon.ai.translate.TranslatorContext;
import software.amazon.ai.util.Utils;

public class SingleShotDetection {

    private static final String ARTIFACT_ID = "ssd";
    private static final MRL SSD =
            new MRL(MRL.Model.CV.OBJECT_DETECTION, ModelZoo.GROUP_ID, ARTIFACT_ID);

    private Repository repository;
    private Metadata metadata;

    public SingleShotDetection(Repository repository) {
        this.repository = repository;
    }

    private void locateMetadata() throws ModelNotFoundException, IOException {
        metadata = repository.locate(SSD);
        if (metadata == null) {
            throw new ModelNotFoundException("SingleShotDetection models not found.");
        }
    }

    public ZooModel<BufferedImage, List<DetectedObject>> loadModel(Map<String, String> criteria)
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
        Model model = Model.newInstance();
        model.load(modelPath, artifact.getName());
        return new ZooModel<>(
                model,
                new ImageTranslator<List<DetectedObject>>() {

                    private static final float THRESHOLD = 0.2f;

                    @Override
                    public NDList processInput(TranslatorContext ctx, BufferedImage input) {
                        // TODO: avoid hard code image size and threshold
                        input = Images.resizeImage(input, 512, 512);
                        return super.processInput(ctx, input);
                    }

                    @Override
                    public List<DetectedObject> processOutput(TranslatorContext ctx, NDList list)
                            throws IOException {
                        Model model = ctx.getModel();
                        List<String> classes = model.getArtifact("classes.txt", Utils::readLines);

                        float[] classIds = list.get(0).toFloatArray();
                        float[] probabilities = list.get(1).toFloatArray();
                        NDArray boundingBoxes = list.get(2);

                        List<DetectedObject> ret = new ArrayList<>();

                        for (int i = 0; i < classIds.length; ++i) {
                            int classId = (int) classIds[i];
                            float probability = probabilities[i];
                            if (classId > 0 && probability > THRESHOLD) {
                                if (classId >= classes.size()) {
                                    throw new AssertionError("Unexpected index: " + classId);
                                }
                                String className = classes.get(classId);
                                float[] box = boundingBoxes.get(0, i).toFloatArray();
                                double x = box[0] / 512;
                                double y = box[1] / 512;
                                double w = box[2] / 512 - x;
                                double h = box[3] / 512 - y;

                                Rectangle rect = new Rectangle(x, y, w, h);
                                ret.add(new DetectedObject(className, probability, rect));
                            }
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
        return list.get(list.size() - 1);
    }

    public List<Artifact> search(Map<String, String> criteria) {
        return metadata.search(VersionRange.parse("0.0.2"), criteria);
    }
}

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
package org.apache.mxnet.zoo.cv.segmentation;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import software.amazon.ai.Model;
import software.amazon.ai.modality.cv.DetectedObject;
import software.amazon.ai.modality.cv.Images;
import software.amazon.ai.modality.cv.Mask;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.translate.Translator;
import software.amazon.ai.translate.TranslatorContext;
import software.amazon.ai.util.Utils;

public class InstanceSegementationTranslator
        implements Translator<BufferedImage, List<DetectedObject>> {

    private static final float THRESHOLD = 0.3f;
    private static final int SHORT_EDGE = 600;
    private static final int MAX_EDGE = 1000;
    private int rescaledWidth;
    private int rescaledHeight;

    @Override
    public NDList processInput(TranslatorContext ctx, BufferedImage image) {
        image = resizeShort(image);
        rescaledWidth = image.getWidth();
        rescaledHeight = image.getHeight();
        Shape shape = new Shape(3, image.getHeight(), image.getWidth());
        DataDesc dataDesc = new DataDesc(shape);
        NDManager manager = ctx.getNDManager();
        FloatBuffer buffer = Images.toFloatBuffer(manager, image);
        NDArray array = manager.create(dataDesc);
        array.set(buffer);

        return new NDList(normalize(array.div(255)));
    }

    @Override
    public List<DetectedObject> processOutput(TranslatorContext ctx, NDList list)
            throws IOException {
        Model model = ctx.getModel();
        List<String> classes = model.getArtifact("classes.txt", Utils::readLines);

        float[] ids = list.get(0).toFloatArray();
        float[] scores = list.get(1).toFloatArray();
        NDArray boundingBoxes = list.get(2);
        NDArray masks = list.get(3);

        List<DetectedObject> result = new ArrayList<>();

        for (int i = 0; i < ids.length; ++i) {
            int classId = (int) ids[i];
            float probability = scores[i];
            if (classId >= 0 && probability > THRESHOLD) {
                if (classId >= classes.size()) {
                    throw new AssertionError("Unexpected index: " + classId);
                }
                String className = classes.get(classId);
                float[] box = boundingBoxes.get(0, i).toFloatArray();
                double x = box[0] / rescaledWidth;
                double y = box[1] / rescaledHeight;
                double w = box[2] / rescaledWidth - x;
                double h = box[3] / rescaledHeight - y;

                Shape maskShape = masks.get(0, i).getShape();
                float[][] maskVal = new float[(int) maskShape.get(0)][(int) maskShape.get(1)];
                float[] flattened = masks.get(0, i).toFloatArray();

                for (int j = 0; j < flattened.length; j++) {
                    maskVal[j / maskVal.length][j % maskVal.length] = flattened[j];
                }

                Mask mask = new Mask(x, y, w, h, maskVal);

                result.add(new DetectedObject(className, probability, mask));
            }
        }
        return result;
    }

    /**
     * resize the image based on the shorter edge or maximum edge length.
     *
     * @param img the input image
     * @return resized image
     */
    private BufferedImage resizeShort(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();
        int min = Math.min(width, height);
        int max = Math.max(width, height);
        float scale = SHORT_EDGE / (float) min;
        if (Math.round(scale * max) > MAX_EDGE) {
            scale = MAX_EDGE / (float) max;
        }
        width = Math.round(width * scale);
        height = Math.round(height * scale);

        return Images.resizeImage(img, width, height);
    }

    private NDArray normalize(NDArray array) {
        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std = {0.229f, 0.224f, 0.225f};
        return Images.normalize(array, mean, std);
    }
}

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mxnet;

import com.amazon.ai.Context;
import com.amazon.ai.image.BoundingBox;
import com.amazon.ai.image.Images;
import com.amazon.ai.image.Rectangle;
import com.amazon.ai.inference.DetectedObject;
import com.amazon.ai.inference.ImageTransformer;
import com.amazon.ai.inference.ObjectDetector;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import com.amazon.ai.ndarray.types.Shape;
import com.sun.jna.Native;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import org.apache.mxnet.engine.MxModel;
import org.apache.mxnet.engine.ResourceAllocator;
import org.apache.mxnet.jna.JnaException;
import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@SuppressWarnings("PMD.SystemPrintln")
public final class SSDClassifierExample {

    private static final Logger logger = LoggerFactory.getLogger(SSDClassifierExample.class);

    private SSDClassifierExample() {}

    private static List<DetectedObject> runObjectDetectionSingle(
            String pathPrefix, String inputImagePath, Context context) throws IOException {
        Shape inputShape = new Shape(1, 3, 224, 224);
        DataDesc dataDesc = new DataDesc(context, "data", inputShape, DataType.FLOAT32, "NCHW");
        BufferedImage img = Images.loadImageFromFile(new File(inputImagePath));

        try (ResourceAllocator alloc = new ResourceAllocator()) {
            MxModel model = MxModel.loadSavedModel(alloc, pathPrefix, 0);
            ImageTransformer<List<DetectedObject>> transformer = new SampleTransformer(dataDesc);
            ObjectDetector objDet = new ObjectDetector(model, transformer);

            return objDet.detect(img);
        }
    }

    public static void main(String[] args) {
        Native.setProtected(true);
        if (!Native.isProtected()) {
            System.out.println("Protection not supported.");
        }
        long init = System.nanoTime();
        System.out.println("Loading native library: " + JnaUtils.getVersion());
        long loaded = System.nanoTime();
        System.out.printf("loadlibrary = %.3f ms.%n", (loaded - init) / 1000000f);

        String userHome = System.getProperty("user.home");
        String modelPathPrefix = userHome + "/source/ptest/squeezenet_v1.1";
        String inputImagePath = userHome + "/source/ptest/kitten.jpg";

        Context context = Context.defaultContext();

        try {
            Shape inputShape = new Shape(1, 3, 224, 224);

            StringBuilder outputStr = new StringBuilder("\n");

            List<DetectedObject> output =
                    runObjectDetectionSingle(modelPathPrefix, inputImagePath, context);

            for (DetectedObject i : output) {
                dumpOutput(i, outputStr, inputShape);
            }
            logger.info(outputStr.toString());
        } catch (IOException | JnaException e) {
            logger.error("", e);
            System.exit(-1);
        }
        System.exit(0);
    }

    private static void dumpOutput(DetectedObject output, StringBuilder sb, Shape shape) {
        int width = shape.get(2);
        int height = shape.get(3);

        sb.append("Class: ")
                .append(output.getClassName())
                .append("\nProbability: ")
                .append(output.getProbability())
                .append("\nCoordinate: ");
        BoundingBox boundingBox = output.getBoundingBox();
        if (boundingBox != null) {
            Rectangle rect = boundingBox.getBounds();
            List<Double> coord =
                    Arrays.asList(
                            rect.getX() * width,
                            rect.getY() * height,
                            (rect.getX() + rect.getWidth()) * width,
                            (rect.getY() + rect.getHeight()) * height);
            boolean first = true;
            for (double c : coord) {
                if (first) {
                    first = false;
                } else {
                    sb.append(", ");
                }
                sb.append(c);
            }
        }
        sb.append('\n');
    }

    private static final class SampleTransformer extends ImageTransformer<List<DetectedObject>> {

        public SampleTransformer(DataDesc dataDesc) {
            super(dataDesc);
        }

        @Override
        public List<DetectedObject> processOutput(NDArray array) {
            return null;
        }
    }
}

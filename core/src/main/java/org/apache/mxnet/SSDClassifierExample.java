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

import com.sun.jna.Native;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.mxnet.image.Image;
import org.apache.mxnet.inferernce.ObjectDetector;
import org.apache.mxnet.inferernce.ObjectDetectorOutput;
import org.apache.mxnet.jna.JnaException;
import org.apache.mxnet.model.DataDesc;
import org.apache.mxnet.model.MxModel;
import org.apache.mxnet.model.ResourceAllocator;
import org.apache.mxnet.model.Shape;
import org.apache.mxnet.types.DataType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class SSDClassifierExample {

    private static final Logger logger = LoggerFactory.getLogger(SSDClassifierExample.class);

    private SSDClassifierExample() {}

    private static List<ObjectDetectorOutput> runObjectDetectionSingle(
            String pathPrefix, String inputImagePath, Context context) throws IOException {
        Shape inputShape = new Shape(1, 3, 224, 224);
        DataDesc dataDesc = new DataDesc("data", inputShape, DataType.FLOAT32, "NCHW");
        BufferedImage img = Image.loadImageFromFile(inputImagePath);

        try (ResourceAllocator alloc = new ResourceAllocator()) {
            MxModel model = MxModel.loadSavedModel(alloc, pathPrefix, 0);
            ObjectDetector objDet = new ObjectDetector(alloc, context, model, dataDesc);

            return objDet.detect(img, 7);
        }
    }

    private static List<List<List<ObjectDetectorOutput>>> runObjectDetectionBatch(
            String pathPrefix, String inputImageDir, Context context) throws IOException {
        Shape inputShape = new Shape(1, 3, 512, 512);
        DataDesc dataDesc = new DataDesc("data", inputShape, DataType.FLOAT32, "NCHW");

        try (ResourceAllocator alloc = new ResourceAllocator()) {
            MxModel model = MxModel.loadSavedModel(alloc, pathPrefix, 0);
            ObjectDetector objDet = new ObjectDetector(alloc, context, model, dataDesc);

            // Loading batch of images from the directory path
            List<List<String>> batchFiles = generateBatches(inputImageDir, 20);
            List<List<List<ObjectDetectorOutput>>> outputList = new ArrayList<>();

            for (List<String> batchFile : batchFiles) {
                List<BufferedImage> imgList = Image.loadInputBatch(batchFile);
                // Running inference on batch of images loaded in previous step
                List<List<ObjectDetectorOutput>> tmp = objDet.detect(imgList, 5);
                outputList.add(tmp);
            }
            return outputList;
        }
    }

    private static List<List<String>> generateBatches(String inputImageDirPath, int batchSize) {
        File dir = new File(inputImageDirPath);

        List<List<String>> output = new ArrayList<>();
        List<String> batch = new ArrayList<>();
        File[] files = dir.listFiles();
        if (files != null) {
            for (File imgFile : files) {
                batch.add(imgFile.getPath());
                if (batch.size() == batchSize) {
                    output.add(batch);
                    batch = new ArrayList<>();
                }
            }
        }
        if (batch.size() > 0) {
            output.add(batch);
        }
        return output;
    }

    public static void main(String[] args) {
        Native.setProtected(true);
        if (!Native.isProtected()) {
            System.out.println("Protection not supported.");
        }
        System.setProperty("jna.library.path", "/Users/lufen/source/mxnet/lib");
        // System.setProperty("jna.library.path", "/usr/local/lib/python2.7/site-packages/mxnet");

        String modelPathPrefix = "/Users/lufen/source/ptest/squeezenet_v1.1";
        String inputImagePath = "/Users/lufen/sample/kitten.jpg";
        String inputDir = "/Users/lufen/sample/";

        Context context = Context.getDefaultContext();

        try {
            Shape inputShape = new Shape(1, 3, 224, 224);

            StringBuilder outputStr = new StringBuilder("\n");

            List<ObjectDetectorOutput> output =
                    runObjectDetectionSingle(modelPathPrefix, inputImagePath, context);

            for (ObjectDetectorOutput i : output) {
                dumpOutput(i, outputStr, inputShape);
            }
            logger.info(outputStr.toString());
        } catch (IOException | JnaException e) {
            logger.error("", e);
            System.exit(-1);
        }
        System.exit(0);
    }

    private static void dumpOutput(ObjectDetectorOutput output, StringBuilder sb, Shape shape) {
        int width = shape.get(2);
        int height = shape.get(3);

        sb.append("Class: ")
                .append(output.getClassName())
                .append('\n')
                .append("Probability: ")
                .append(output.getProbability())
                .append('\n')
                .append("Coordinate: ");
        List<Float> coord =
                Arrays.asList(
                        output.getXMin() * width,
                        output.getXMax() * height,
                        output.getYMin() * width,
                        output.getYMax() * height);
        boolean first = true;
        for (float c : coord) {
            if (first) {
                first = false;
            } else {
                sb.append(", ");
            }
            sb.append(c);
        }
        sb.append('\n');
    }
}

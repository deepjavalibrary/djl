/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package org.apache.mxnet.inferernce;

import java.awt.image.BufferedImage;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.mxnet.Context;
import org.apache.mxnet.image.Image;
import org.apache.mxnet.model.DataDesc;
import org.apache.mxnet.model.Module;
import org.apache.mxnet.model.MxModel;
import org.apache.mxnet.model.NdArray;
import org.apache.mxnet.model.ResourceAllocator;
import org.apache.mxnet.model.Shape;

public class ObjectDetector implements Predictor {

    private ResourceAllocator alloc;
    private Context context;
    private MxModel model;
    private DataDesc dataDescriptor;
    private Module module;
    private boolean initialized;

    public ObjectDetector(
            ResourceAllocator alloc, Context context, MxModel model, DataDesc dataDescriptor) {
        this.alloc = alloc;
        this.context = context;
        this.model = model;
        this.dataDescriptor = dataDescriptor;
    }

    public synchronized void initialize() {
        if (initialized) {
            return;
        }

        List<DataDesc> descriptors = Collections.singletonList(dataDescriptor);
        Module.Builder builder = Module.forInference(context, model, descriptors);
        module = builder.build(alloc);

        initialized = true;
    }

    public List<ObjectDetectorOutput> detect(BufferedImage inputImage, int topK) {
        List<BufferedImage> images = Collections.singletonList(inputImage);
        List<List<ObjectDetectorOutput>> results = detect(images, topK);
        if (results.isEmpty()) {
            return Collections.emptyList();
        }
        return results.get(0);
    }

    public List<List<ObjectDetectorOutput>> detect(List<BufferedImage> inputBatch, int topK) {
        List<List<ObjectDetectorOutput>> ret = new ArrayList<>(inputBatch.size());
        try (NdArray ndArray = imageToNdArray(inputBatch)) {
            NdArray[] result = predict(ndArray);
            assert result.length == inputBatch.size();

            for (NdArray array : result) {
                ret.add(toOutput(array, topK));
            }
        }

        return ret;
    }

    private NdArray imageToNdArray(List<BufferedImage> inputBatch) {
        int width = dataDescriptor.getShape('W');
        int height = dataDescriptor.getShape('H');
        Shape shape = dataDescriptor.getShape();

        NdArray ndArray = new NdArray(alloc, context, shape);
        boolean first = true;
        for (BufferedImage img : inputBatch) {
            BufferedImage image = Image.reshapeImage(img, width, height);
            FloatBuffer data = Image.toDirectBuffer(image);
            if (first) {
                first = false;
            } else {
                Map<String, Object> args = new HashMap<>();
                args.put("aixs", 0);
                ndArray.genericNDArrayFunctionInvoke("expand_dims", args);
            }
            ndArray.set(data);
        }
        return ndArray;
    }

    private List<ObjectDetectorOutput> toOutput(NdArray ndArray, int topK) {
        int length = ndArray.getShape().head();
        length = Math.min(length, topK);
        List<ObjectDetectorOutput> ret = new ArrayList<>(length);
        try (NdArray nd = ndArray.at(0)) {
            NdArray sorted = nd.argsort(-1, false);
            NdArray top = sorted.slice(0, topK);

            float[] probabilities = nd.toFloatArray();
            float[] indices = top.toFloatArray();

            sorted.close();
            top.close();

            for (int i = 0; i < topK; ++i) {
                int index = (int) indices[i];
                String className = model.getSynset()[index];
                float[] args = new float[5];
                args[0] = probabilities[index];
                ObjectDetectorOutput output = new ObjectDetectorOutput(className, args);
                ret.add(output);
            }
        }
        return ret;
    }

    @Override
    public NdArray[] predict(NdArray... ndArrays) {
        if (module == null) {
            List<DataDesc> descriptors = Collections.singletonList(dataDescriptor);
            Module.Builder builder = new Module.Builder(context, model, descriptors, false);
            module = builder.build(alloc);
        }
        module.forward(ndArrays);

        NdArray[] ret = module.getOutputs();
        for (NdArray nd : ret) {
            nd.waitToRead();
        }
        return ret;
    }
}

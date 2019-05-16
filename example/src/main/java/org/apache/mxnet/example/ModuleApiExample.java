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

package org.apache.mxnet.example;

import com.amazon.ai.Context;
import com.amazon.ai.example.util.AbstractExample;
import com.amazon.ai.example.util.LogUtils;
import com.amazon.ai.image.Images;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.Shape;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.apache.mxnet.engine.Module;
import org.apache.mxnet.engine.MxModel;
import org.apache.mxnet.engine.MxNDFactory;
import org.slf4j.Logger;

public final class ModuleApiExample extends AbstractExample {

    private static final Logger logger = LogUtils.getLogger(ModuleApiExample.class);

    private ModuleApiExample() {}

    public static void main(String[] args) {
        new ModuleApiExample().runExample(args);
    }

    @Override
    public void predict(String modelDir, String modelName, BufferedImage img, int iteration)
            throws IOException {
        Context context = Context.defaultContext();

        String modelPathPrefix = modelDir + '/' + modelName;

        BufferedImage image = Images.reshapeImage(img, 224, 224);
        FloatBuffer data = Images.toFloatBuffer(image);

        try (MxNDFactory factory = MxNDFactory.SYSTEM_FACTORY.newSubFactory()) {
            MxModel model = MxModel.loadModel(factory, modelPathPrefix, 0);
            DataDesc dataDesc = new DataDesc(new Shape(1, 3, 224, 224), "data");
            model.setDataNames(dataDesc);

            long init = System.nanoTime();
            Module.Builder builder = new Module.Builder(context, model, false);
            Module module = builder.build();
            long loadModel = System.nanoTime();
            logger.info(String.format("bind model  = %.3f ms.", (loadModel - init) / 1000000f));

            List<Long> inferenceTime = new ArrayList<>(iteration);
            for (int i = 0; i < iteration; ++i) {
                long begin = System.nanoTime();

                NDArray ndArray = factory.create(dataDesc);
                ndArray.set(data);

                NDList input = new NDList(ndArray);
                module.forward(input);

                NDList ret = module.getOutputs();
                for (NDArray nd : ret) {
                    nd.waitAll();
                }

                long inference = System.nanoTime();
                inferenceTime.add(inference - begin);

                NDArray sorted = ret.get(0).argsort(-1, false);
                NDArray top = sorted.slice(0, 1);

                if (i == 0) {
                    float[] indices = top.toFloatArray();
                    String className = model.getSynset()[(int) indices[0]];
                    logger.info(String.format("Result: %s", className));
                }
            }

            Collections.sort(inferenceTime);

            float p50 = inferenceTime.get(iteration / 2) / 1000000f;
            float p90 = inferenceTime.get(iteration * 9 / 10) / 1000000f;

            logger.info(String.format("inference P50: %.3f ms, P90: %.3f ms", p50, p90));
        }
    }
}

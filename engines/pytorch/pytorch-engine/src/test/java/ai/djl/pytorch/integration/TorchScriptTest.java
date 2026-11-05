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

package ai.djl.pytorch.integration;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtIValueNDArray;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtSymbolBlock;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.testing.TestRequirements;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.PairList;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

/** The file is for integration test for special TorchScript input. */
public class TorchScriptTest {

    @Test
    public void testDictInput() throws ModelException, IOException, TranslateException {
        TestRequirements.notMacX86();

        try (NDManager manager = NDManager.newBaseManager()) {
            Criteria<NDList, NDList> criteria =
                    Criteria.builder()
                            .setTypes(NDList.class, NDList.class)
                            .optModelUrls("https://resources.djl.ai/test-models/dict_input.zip")
                            .optProgress(new ProgressBar())
                            .build();

            Path modelFile;
            try (ZooModel<NDList, NDList> model = criteria.loadModel();
                    Predictor<NDList, NDList> predictor = model.newPredictor()) {
                PtNDArray array = (PtNDArray) manager.ones(new Shape(2, 2));
                array.setName("input1.input");
                NDList output = predictor.predict(new NDList(array));
                Assert.assertEquals(output.singletonOrThrow(), array);

                modelFile = model.getModelPath().resolve(model.getName() + ".pt");
            }

            criteria =
                    Criteria.builder()
                            .setTypes(NDList.class, NDList.class)
                            .optModelPath(modelFile)
                            .optProgress(new ProgressBar())
                            .build();

            try (ZooModel<NDList, NDList> model = criteria.loadModel()) {
                Assert.assertEquals(model.getName(), "dict_input");
            }
        }
    }

    @Test
    public void testDictInputCarrier() throws ModelException, IOException, TranslateException {
        TestRequirements.notMacX86();

        NoBatchifyTranslator<NDList, NDList> translator =
                new NoBatchifyTranslator<NDList, NDList>() {
                    @Override
                    public NDList processOutput(TranslatorContext ctx, NDList list) {
                        return list;
                    }

                    @Override
                    public NDList processInput(TranslatorContext ctx, NDList input) {
                        return input;
                    }
                };
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls("https://resources.djl.ai/test-models/dict_input.zip")
                        .optTranslator(translator)
                        .optProgress(new ProgressBar())
                        .build();

        try (NDManager manager = NDManager.newBaseManager();
                ZooModel<NDList, NDList> model = criteria.loadModel();
                Predictor<NDList, NDList> predictor = model.newPredictor()) {
            // The published model unpacks exactly one dict value, so multi-key/multi-dtype
            // coverage lives in IValueStringMapBlobsTest/IValueUtilsTest.
            ByteBuffer parent = ByteBuffer.allocateDirect(16).order(ByteOrder.nativeOrder());
            ByteBuffer floats = slice(parent, 0, 16);
            for (int i = 0; i < 4; ++i) {
                floats.putFloat(i * 4, 1f);
            }
            PtIValueNDArray carrier =
                    ((PtNDManager) manager)
                            .createStringMap(
                                    "input1",
                                    new String[] {"input"},
                                    new ByteBuffer[] {floats},
                                    new long[][] {{2, 2}},
                                    new DataType[] {DataType.FLOAT32});
            NDArray expected = manager.ones(new Shape(2, 2));
            for (int i = 0; i < 2; ++i) {
                NDList output = predictor.predict(new NDList(carrier));
                Assert.assertEquals(output.singletonOrThrow(), expected);
            }

            PairList<String, Shape> desc = ((PtSymbolBlock) model.getBlock()).describeInput();
            Assert.assertEquals(desc.size(), 1);
            Assert.assertEquals(desc.get("input1.input"), new Shape(2, 2));
        }
    }

    private static ByteBuffer slice(ByteBuffer parent, int offset, int length) {
        ByteBuffer dup = parent.duplicate();
        dup.position(offset);
        dup.limit(offset + length);
        return dup.slice().order(ByteOrder.nativeOrder());
    }

    @Test
    public void testInputOutput() throws IOException, ModelException {
        TestRequirements.notMacX86();

        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls("djl://ai.djl.pytorch/resnet/0.0.1/traced_resnet18")
                        .optProgress(new ProgressBar())
                        .build();
        try (ZooModel<NDList, NDList> model = criteria.loadModel()) {
            PtNDManager manager = (PtNDManager) model.getNDManager();
            Path modelFile = model.getModelPath().resolve("traced_resnet18.pt");
            try (InputStream is = Files.newInputStream(modelFile)) {
                PtSymbolBlock block = JniUtils.loadModule(manager, is, true, false);
                ByteArrayOutputStream os = new ByteArrayOutputStream();
                JniUtils.writeModule(block, os, true);
                ByteArrayInputStream bis = new ByteArrayInputStream(os.toByteArray());
                JniUtils.loadModule(manager, bis, true, true);
                bis.close();
                os.close();
            }
        }
    }

    @Test
    public void testGetMethodNames() throws ModelException, IOException {
        TestRequirements.notMacX86();

        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls("djl://ai.djl.pytorch/resnet/0.0.1/traced_resnet18")
                        .optProgress(new ProgressBar())
                        .build();
        try (ZooModel<NDList, NDList> model = criteria.loadModel()) {
            PtSymbolBlock block = (PtSymbolBlock) model.getBlock();
            String[] result = JniUtils.getMethodNames(block);
            Assert.assertEquals(result[0], "forward");
        }
    }
}

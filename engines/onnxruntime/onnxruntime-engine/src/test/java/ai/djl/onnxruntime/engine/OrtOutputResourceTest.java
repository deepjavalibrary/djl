/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.onnxruntime.engine;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.onnxruntime.OnnxSequence;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Collections;
import java.util.Map;

public class OrtOutputResourceTest {

    @Test
    public void testMapSequenceClosedAfterConversion()
            throws OrtException, ReflectiveOperationException {
        OrtEngine engine = (OrtEngine) Engine.getEngine("OnnxRuntime");
        OrtEnvironment environment = engine.getEnv();
        try (OrtNDManager manager = (OrtNDManager) engine.newBaseManager();
                OrtSession.SessionOptions options = new OrtSession.SessionOptions();
                OnnxTensor input =
                        OnnxTensor.createTensor(
                                environment, new float[][] {{0.25f, 0.75f}, {0.5f, 0.5f}})) {
            OrtSession session =
                    environment.createSession(
                            "src/test/resources/sequence-output/maps.onnx", options);
            OrtSymbolBlock block = new OrtSymbolBlock(session, manager);
            OrtSession.Result result = session.run(Collections.singletonMap("input", input));
            try {
                OnnxSequence sequence = (OnnxSequence) result.get(0);
                NDList output = evaluateOutput(block, result);
                Assert.assertTrue(
                        sequence.isClosed(), "Converted sequence must release its handle");
                Assert.assertEquals(output.singletonOrThrow().getShape(), new Shape(2, 2));
                Assert.assertEquals(
                        output.singletonOrThrow().toFloatArray(),
                        new float[] {0.25f, 0.75f, 0.5f, 0.5f});
            } finally {
                closeUnownedValues(result);
            }
        }
    }

    @Test
    public void testTensorSequenceClosedWithoutClosingOutputTensors()
            throws OrtException, ReflectiveOperationException {
        OrtEngine engine = (OrtEngine) Engine.getEngine("OnnxRuntime");
        OrtEnvironment environment = engine.getEnv();
        OnnxTensor direct;
        OnnxTensor child;
        try (OrtNDManager manager = (OrtNDManager) engine.newBaseManager();
                OrtSession.SessionOptions options = new OrtSession.SessionOptions();
                OnnxTensor input = OnnxTensor.createTensor(environment, new float[] {2f, 3f})) {
            OrtSession session =
                    environment.createSession(
                            "src/test/resources/sequence-output/tensors.onnx", options);
            OrtSymbolBlock block = new OrtSymbolBlock(session, manager);
            OrtSession.Result result = session.run(Collections.singletonMap("input", input));
            try {
                OnnxSequence sequence = (OnnxSequence) result.get("sequence").get();
                direct = (OnnxTensor) result.get("direct").get();
                NDList output = evaluateOutput(block, result);
                Assert.assertTrue(
                        sequence.isClosed(), "Converted sequence must release its handle");
                Assert.assertFalse(direct.isClosed(), "Tensor ownership belongs to the NDManager");
                Assert.assertEquals(output.size(), 2);
                for (int i = 0; i < output.size(); ++i) {
                    Assert.assertEquals(output.get(i).toFloatArray(), new float[] {2f, 3f});
                }
                child = ((OrtNDArray) output.get(0)).getTensor();
                Assert.assertNotSame(child, direct);
            } finally {
                closeUnownedValues(result);
            }
        }
        Assert.assertTrue(child.isClosed(), "NDManager must own the extracted tensor");
        Assert.assertTrue(direct.isClosed(), "NDManager must own the normal tensor");
    }

    @Test
    public void testMapSequenceClosedWhenConversionFails()
            throws OrtException, ReflectiveOperationException {
        OrtEngine engine = (OrtEngine) Engine.getEngine("OnnxRuntime");
        OrtEnvironment environment = engine.getEnv();
        try (OrtNDManager manager = (OrtNDManager) engine.newBaseManager();
                OrtSession.SessionOptions options = new OrtSession.SessionOptions();
                OnnxTensor input =
                        OnnxTensor.createTensor(
                                environment, new float[][] {{0.25f, 0.75f}, {0.5f, 0.5f}})) {
            OrtSession session =
                    environment.createSession(
                            "src/test/resources/sequence-output/maps.onnx", options);
            OrtSymbolBlock block = new OrtSymbolBlock(session, manager);
            OrtSession.Result result = session.run(Collections.singletonMap("input", input));
            try {
                OnnxSequence sequence = (OnnxSequence) result.get(0);
                manager.cap();
                InvocationTargetException error =
                        Assert.expectThrows(
                                InvocationTargetException.class,
                                () -> evaluateOutput(block, result));
                Assert.assertTrue(error.getCause() instanceof IllegalStateException);
                Assert.assertTrue(
                        sequence.isClosed(), "Failed conversion must release its sequence");
            } finally {
                closeUnownedValues(result);
            }
        }
    }

    private NDList evaluateOutput(OrtSymbolBlock block, OrtSession.Result result)
            throws ReflectiveOperationException {
        Method method =
                OrtSymbolBlock.class.getDeclaredMethod("evaluateOutput", OrtSession.Result.class);
        method.setAccessible(true);
        return (NDList) method.invoke(block, result);
    }

    private void closeUnownedValues(OrtSession.Result result) {
        for (Map.Entry<String, OnnxValue> entry : result) {
            OnnxValue value = entry.getValue();
            if (value instanceof OnnxSequence && !value.isClosed()) {
                value.close();
            }
        }
    }
}

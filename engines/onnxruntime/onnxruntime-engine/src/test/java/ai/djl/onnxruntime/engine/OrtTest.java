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
package ai.djl.onnxruntime.engine;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.onnxruntime.zoo.tabular.softmax_regression.IrisFlower;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.util.Utils;
import ai.onnxruntime.OrtException;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class OrtTest {

    @BeforeClass
    public void setUp() {
        System.setProperty("ai.djl.onnxruntime.num_threads", "1");
        System.setProperty("ai.djl.onnxruntime.num_interop_threads", "1");
    }

    @Test
    public void testOrtVersion() throws IOException {
        Engine engine = Engine.getEngine("OnnxRuntime");
        Path path = Paths.get("../../../gradle/libs.versions.toml");
        String version = null;
        Pattern pattern = Pattern.compile("onnxruntime = \"([\\d.]+)\"");
        for (String line : Utils.readLines(path)) {
            Matcher m = pattern.matcher(line);
            if (m.matches()) {
                version = m.group(1);
                break;
            }
        }
        Assert.assertEquals(engine.getVersion(), version);
    }

    @Test
    public void testOrt() throws TranslateException, ModelException, IOException {
        try {
            Criteria<IrisFlower, Classifications> criteria =
                    Criteria.builder()
                            .setTypes(IrisFlower.class, Classifications.class)
                            .optModelUrls("djl://ai.djl.onnxruntime/iris_flowers")
                            .optEngine("OnnxRuntime") // use OnnxRuntime engine
                            .optOption("interOpNumThreads", "1")
                            .optOption("intraOpNumThreads", "1")
                            .optOption("executionMode", "SEQUENTIAL")
                            .optOption("optLevel", "NO_OPT")
                            .optOption("memoryPatternOptimization", "true")
                            .optOption("cpuArenaAllocator", "true")
                            .optOption("disablePerSessionThreads", "true")
                            .optOption("profilerOutput", "build/testOrtProfiling")
                            .build();

            IrisFlower virginica = new IrisFlower(1.0f, 2.0f, 3.0f, 4.0f);
            try (ZooModel<IrisFlower, Classifications> model = criteria.loadModel();
                    Predictor<IrisFlower, Classifications> predictor = model.newPredictor()) {
                Classifications classifications = predictor.predict(virginica);
                Assert.assertEquals(classifications.best().getClassName(), "virginica");

                Model m = Model.newInstance("model", "OnnxRuntime");
                Path path = model.getModelPath();
                Assert.assertThrows(() -> m.load(path, "invalid.onnx"));

                m.load(path, null);
                m.close();

                Model m2 = Model.newInstance("model", "OnnxRuntime");
                Path modelFile = path.resolve(model.getName() + ".onnx");
                m2.load(modelFile);
                m2.close();

                // Test load model from stream
                Model stream = Model.newInstance("model", "OnnxRuntime");
                try (InputStream is = Files.newInputStream(modelFile)) {
                    stream.load(is);
                }
                stream.close();
            }
        } catch (UnsatisfiedLinkError e) {
            /*
             * FIXME: Ort requires libgomp.so.1 pre-installed, we should manually copy
             * libgomp.so.1 to our cache folder and set "onnxruntime.native." + library + ".path"
             * to djl cache directory.
             */
            throw new SkipException("Ignore missing libgomp.so.1 error.");
        }
    }

    @Test
    public void testNDArray() throws OrtException {
        try (NDManager manager = OrtNDManager.getSystemManager().newSubManager()) {
            NDArray zeros = manager.zeros(new Shape(1, 2));
            float[] data = zeros.toFloatArray();
            Assert.assertEquals(data[0], 0);

            NDArray ones = manager.ones(new Shape(1, 2));
            data = ones.toFloatArray();
            Assert.assertEquals(data[0], 1);

            float[] buf = {0f, 1f, 2f, 3f};
            NDArray array = manager.create(buf);
            Assert.assertEquals(array.toFloatArray(), buf);

            array = manager.create("string");
            Assert.assertEquals(array.toStringArray()[0], "string");
            final NDArray a = array;
            Assert.assertThrows(IllegalArgumentException.class, a::toByteBuffer);

            array = manager.create(new String[] {"string1", "string2"});
            Assert.assertEquals(array.toStringArray()[1], "string2");

            float[][] value = (float[][]) ((OrtNDArray) ones).getTensor().getValue();
            Assert.assertEquals(value[0], new float[] {1, 1});

            array = manager.create(new Shape(1), DataType.BOOLEAN);
            Assert.assertEquals(array.getDataType(), DataType.BOOLEAN);

            array = manager.create(new Shape(1), DataType.FLOAT16);
            Assert.assertEquals(array.getDataType(), DataType.FLOAT16);

            array = manager.create(new Shape(1), DataType.BFLOAT16);
            Assert.assertEquals(array.getDataType(), DataType.BFLOAT16);

            array = manager.create(new double[] {0});
            Assert.assertEquals(array.getDataType(), DataType.FLOAT64);

            array = manager.create(new Shape(1), DataType.FLOAT64);
            Assert.assertEquals(array.getDataType(), DataType.FLOAT64);

            array = manager.create(new Shape(1), DataType.INT8);
            Assert.assertEquals(array.getDataType(), DataType.INT8);

            array = manager.create(new Shape(1), DataType.UINT8);
            Assert.assertEquals(array.getDataType(), DataType.UINT8);

            array = manager.create(new int[] {0});
            Assert.assertEquals(array.getDataType(), DataType.INT32);

            array = manager.create(new Shape(1), DataType.INT32);
            Assert.assertEquals(array.getDataType(), DataType.INT32);

            array = manager.create(new long[] {0L});
            Assert.assertEquals(array.getDataType(), DataType.INT64);

            array = manager.create(new Shape(1), DataType.INT64);
            Assert.assertEquals(array.getDataType(), DataType.INT64);

            Assert.assertThrows(() -> manager.create(new Shape(0), DataType.FLOAT32));
            Assert.assertThrows(() -> manager.create(new Shape(1), DataType.UINT32));
        }
    }

    @Test
    public void testStringTensor() throws ModelException, IOException, TranslateException {
        setAlternativeEngineDisabled(true);
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optEngine("OnnxRuntime")
                        .optModelUrls(
                                "https://resources.djl.ai/test-models/onnxruntime/pipeline_tfidf.zip")
                        .build();
        try (ZooModel<NDList, NDList> model = criteria.loadModel();
                Predictor<NDList, NDList> predictor = model.newPredictor()) {
            OrtNDManager manager = (OrtNDManager) model.getNDManager();
            NDArray stringNd =
                    manager.create(
                            new String[] {" Re: Jack can't hide from keith@cco.", " I like dogs"},
                            new Shape(1, 2));
            NDList result = predictor.predict(new NDList(stringNd));
            Assert.assertEquals(result.size(), 2);
            Assert.assertEquals(result.get(0).toLongArray(), new long[] {1});
        }
        setAlternativeEngineDisabled(false);
    }

    @Test
    public void testAlternativeArray() {
        try (NDManager manager = OrtNDManager.getSystemManager().newSubManager()) {
            NDArray array = manager.zeros(new Shape(1, 2));
            Assert.assertEquals(array.get(0).toFloatArray(), new float[] {0, 0});
        }

        setAlternativeEngineDisabled(true);
        try (NDManager manager = OrtNDManager.getSystemManager().newSubManager()) {
            NDArray array = manager.zeros(new Shape(1, 2));
            Assert.expectThrows(UnsupportedOperationException.class, () -> array.get(0));
        }
        setAlternativeEngineDisabled(false);
    }

    private void setAlternativeEngineDisabled(boolean enable) {
        System.setProperty("ai.djl.onnx.disable_alternative", String.valueOf(enable));
        Engine engine = Engine.getEngine(OrtEngine.ENGINE_NAME);
        try {
            Field field = OrtEngine.class.getDeclaredField("initialized");
            field.setAccessible(true);
            field.setBoolean(engine, false);
            field = OrtEngine.class.getDeclaredField("alternativeEngine");
            field.setAccessible(true);
            field.set(engine, null);
        } catch (ReflectiveOperationException ignore) {
            // ignore
        }
    }
}

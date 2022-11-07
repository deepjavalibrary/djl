/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.ml.xgboost;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.testing.TestRequirements;
import ai.djl.training.util.DownloadUtils;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;

public class XgbModelTest {

    @BeforeClass
    public void downloadXGBoostModel() throws IOException {
        TestRequirements.notArm();
        TestRequirements.notWindows();

        Path modelDir = Paths.get("build/model");
        DownloadUtils.download(
                "https://resources.djl.ai/test-models/xgboost/regression.json",
                modelDir.resolve("regression.json").toString());
    }

    @Test
    public void testVersion() {
        Engine engine = Engine.getEngine("XGBoost");
        Assert.assertEquals("1.7.1", engine.getVersion());
    }

    /*
    This test depends on rapis, which doesn't work for CPU
    @Test
    public void testCudf() {
        Engine engine = Engine.getEngine("XGBoost");
        if (engine.hasCapability(StandardCapabilities.CUDA)) {
            Float[] data = new Float[]{1.f, null, 5.f, 7.f, 9.f};
            Table x = new Table.TestBuilder().column(data).build();
            ColumnBatch columnBatch = new CudfColumnBatch(x, null, null, null);

            try (XgbNDManager manager = (XgbNDManager)engine.newBaseManager()) {
                NDArray array = manager.create(columnBatch,0, 1);
                Assert.assertEquals(array.getShape().get(0), 5);
            }
        }
    }
    */

    @Test
    public void testLoad() throws MalformedModelException, IOException, TranslateException {
        try (Model model = Model.newInstance("XGBoost")) {
            model.load(Paths.get("build/model"), "regression");
            Predictor<NDList, NDList> predictor = model.newPredictor(new NoopTranslator());
            try (NDManager manager = NDManager.newBaseManager()) {
                NDArray array = manager.ones(new Shape(10, 13));
                NDList output = predictor.predict(new NDList(array));
                Assert.assertEquals(output.singletonOrThrow().getDataType(), DataType.FLOAT32);
                Assert.assertEquals(output.singletonOrThrow().toFloatArray().length, 10);
            }
        }
    }

    @Test
    public void testNDArray() {
        try (XgbNDManager manager =
                (XgbNDManager) XgbNDManager.getSystemManager().newSubManager()) {
            manager.setMissingValue(Float.NaN);
            NDArray zeros = manager.zeros(new Shape(1, 2));
            Assert.expectThrows(UnsupportedOperationException.class, zeros::toFloatArray);

            NDArray ones = manager.ones(new Shape(1, 2));
            Assert.expectThrows(UnsupportedOperationException.class, ones::toFloatArray);

            float[] buf = {0f, 1f, 2f, 3f};

            ByteBuffer bb = ByteBuffer.allocateDirect(4 * buf.length);
            bb.asFloatBuffer().put(buf);
            // output only NDArray
            XgbNDArray xgbArray = (XgbNDArray) manager.create(bb, new Shape(4), DataType.FLOAT32);
            Assert.assertEquals(xgbArray.toFloatArray(), buf);
            final XgbNDArray a = xgbArray;
            Assert.assertThrows(UnsupportedOperationException.class, a::getHandle);
            Assert.assertThrows(
                    UnsupportedOperationException.class,
                    () -> manager.create(bb.asFloatBuffer(), new Shape(4)));

            int[] expected = {0, 1, 2, 3};
            bb.rewind();
            bb.asIntBuffer().put(expected);
            xgbArray = (XgbNDArray) manager.create(bb, new Shape(2, 2), DataType.INT32);
            Assert.assertEquals(xgbArray.toIntArray(), expected);
            Assert.assertThrows(
                    UnsupportedOperationException.class,
                    () -> manager.create(bb.asIntBuffer(), new Shape(2, 24), DataType.INT32));

            NDArray array = manager.create(buf, new Shape(2, 2));
            Assert.assertEquals(array.getDataType(), DataType.FLOAT32);

            long[] indptr = {0, 2, 2, 3};
            long[] indices = {0, 2, 1};
            FloatBuffer fb = FloatBuffer.wrap(new float[] {7, 8, 9});
            array = manager.createCSR(fb, indptr, indices, new Shape(3, 4));
            Assert.assertEquals(array.getSparseFormat(), SparseFormat.CSR);
        }
    }
}

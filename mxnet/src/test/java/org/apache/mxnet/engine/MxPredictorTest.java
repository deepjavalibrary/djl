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
package org.apache.mxnet.engine;
// CHECKSTYLE:OFF:AvoidStaticImport

import static org.powermock.api.mockito.PowerMockito.mockStatic;

import com.amazon.ai.Context;
import com.amazon.ai.TranslateException;
import com.amazon.ai.Translator;
import com.amazon.ai.TranslatorContext;
import com.amazon.ai.inference.Predictor;
import com.amazon.ai.metric.Metrics;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.Shape;
import com.amazon.ai.test.MockMxnetLibrary;
import org.apache.mxnet.jna.LibUtils;
import org.apache.mxnet.jna.MxnetLibrary;
import org.powermock.api.mockito.PowerMockito;
import org.powermock.core.classloader.annotations.PrepareForTest;
import org.powermock.modules.testng.PowerMockTestCase;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

// CHECKSTYLE:ON:AvoidStaticImport

@PrepareForTest(LibUtils.class)
public class MxPredictorTest extends PowerMockTestCase {

    private MxnetLibrary library;

    @BeforeClass
    public void prepare() {
        mockStatic(LibUtils.class);
        library = new MockMxnetLibrary();
        PowerMockito.when(LibUtils.loadLibrary()).thenReturn(library);
    }

    @Test
    public void testPredict() throws Exception {
        String prefix = "A";
        int epoch = 122;
        MxModel model = MxModel.loadModel(prefix, epoch);
        DummyTranslator translator = new DummyTranslator();
        Predictor<Integer, NDList> predictor = new MxPredictor<>(model, translator, Context.gpu());
        predictor.setMetrics(new Metrics());
        NDList output = predictor.predict(5);
        Assert.assertEquals(output.size(), 3);
    }

    private static final class DummyTranslator implements Translator<Integer, NDList> {

        public DummyTranslator() {
            super();
        }

        @Override
        public NDList processInput(TranslatorContext ctx, Integer input) throws TranslateException {
            if (input == null) {
                throw new TranslateException("Input is null");
            }
            NDArray nd = ctx.getNDFactory().create(new DataDesc(new Shape(1, 3)));
            return new NDList(nd);
        }

        @Override
        public NDList processOutput(TranslatorContext ctx, NDList list) throws TranslateException {
            if (list.size() == 0) {
                throw new TranslateException("Not output found");
            }
            return list;
        }
    }
}

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

import org.apache.mxnet.jna.LibUtils;
import org.apache.mxnet.jna.MxnetLibrary;
import org.apache.mxnet.test.MockMxnetLibrary;
import org.powermock.api.mockito.PowerMockito;
import org.powermock.core.classloader.annotations.PrepareForTest;
import org.powermock.modules.testng.PowerMockTestCase;
import org.testng.Assert;
import org.testng.IObjectFactory;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.ObjectFactory;
import org.testng.annotations.Test;
import software.amazon.ai.Context;
import software.amazon.ai.inference.Predictor;
import software.amazon.ai.metric.Metrics;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.translate.TranslateException;
import software.amazon.ai.translate.Translator;
import software.amazon.ai.translate.TranslatorContext;

// CHECKSTYLE:ON:AvoidStaticImport

@PrepareForTest(LibUtils.class)
public class MxPredictorTest extends PowerMockTestCase {

    @BeforeClass
    public void prepare() {
        mockStatic(LibUtils.class);
        MxnetLibrary library = new MockMxnetLibrary();
        PowerMockito.when(LibUtils.loadLibrary()).thenReturn(library);
    }

    @Test
    public void testPredict() throws Exception {
        String prefix = "A";
        int epoch = 122;
        MxModel model = MxModel.load(prefix, epoch);
        DummyTranslator translator = new DummyTranslator();
        Predictor<Integer, NDList> predictor = new MxPredictor<>(model, translator, Context.gpu());
        predictor.setMetrics(new Metrics());
        NDList output = predictor.predict(5);
        Assert.assertEquals(output.size(), 3);
    }

    @ObjectFactory
    public IObjectFactory getObjectFactory() {
        return new org.powermock.modules.testng.PowerMockObjectFactory();
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
            NDArray nd = ctx.getNDManager().create(new DataDesc(new Shape(1, 3)));
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

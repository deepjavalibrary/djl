/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.engine.rpc;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.testing.TestServer;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.util.JsonSerializable;
import ai.djl.util.JsonUtils;
import ai.djl.util.Pair;

import com.google.gson.JsonElement;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.lang.reflect.Type;
import java.util.Iterator;
import java.util.Map;

public class RpcTranslatorFactoryTest {

    @Test
    public void testRpcTranslatorFactory() throws TranslateException {
        RpcTranslatorFactory factory = new RpcTranslatorFactory();
        Map<String, String> arguments = Map.of("djl_rpc_uri", "http://localhost");
        Translator<String, String> translator =
                factory.newInstance(String.class, String.class, null, arguments);
        Assert.assertNotNull(translator);

        RpcTranslatorFactory customFactory =
                new RpcTranslatorFactory(
                        new TypeConverter<String, String>() {

                            /** {@inheritDoc} */
                            @Override
                            public Pair<Type, Type> getSupportedType() {
                                return new Pair<>(String.class, String.class);
                            }

                            /** {@inheritDoc} */
                            @Override
                            public Input toInput(String in) {
                                return new Input();
                            }

                            /** {@inheritDoc} */
                            @Override
                            public String fromOutput(Output out) {
                                return "";
                            }
                        });

        translator = customFactory.newInstance(String.class, String.class, null, arguments);
        Assert.assertNotNull(translator);

        Assert.assertThrows(
                () -> customFactory.newInstance(Integer.class, String.class, null, arguments));
    }

    @Test
    public void testRpcRepository() throws ModelException, IOException, TranslateException {
        String content = "{\"data\": \"echo\"}";
        try (TestServer server = TestServer.newInstance(content)) {
            int port = server.getPort();

            Criteria<Input, Output> criteria =
                    Criteria.builder()
                            .setTypes(Input.class, Output.class)
                            .optModelUrls("http://localhost:" + port + "/invocations")
                            .optArgument("method", "POST")
                            .optArgument("API_KEY", "1234")
                            .optArgument("Content-Type", "text/plain")
                            .optArgument("Accept", "*/*")
                            .build();
            try (ZooModel<Input, Output> model = criteria.loadModel();
                    Predictor<Input, Output> predictor = model.newPredictor()) {
                Input in = new Input();
                in.add("Hello, I am a [MASK] model.");
                Output ret = predictor.predict(in);
                Assert.assertEquals(ret.getData().getAsString(), content);
            }

            Criteria<String, String> criteria1 =
                    Criteria.builder()
                            .setTypes(String.class, String.class)
                            .optModelUrls("http://localhost:" + port + "/invocations")
                            .optArgument("method", "POST")
                            .optArgument("API_KEY", "1234")
                            .optArgument("Content-Type", "application/json")
                            .optArgument("Accept", "*/*")
                            .build();
            try (ZooModel<String, String> model = criteria1.loadModel();
                    Predictor<String, String> predictor = model.newPredictor()) {
                String ret = predictor.predict("Hello, I am a [MASK] model.");
                Assert.assertEquals(ret, content);
            }

            TestData testInput = new TestData("Hello");
            Criteria<TestData, TestData> criteria2 =
                    Criteria.builder()
                            .setTypes(TestData.class, TestData.class)
                            .optModelUrls("http://localhost:" + port + "/invocations")
                            .optArgument("method", "POST")
                            .optArgument("API_KEY", "1234")
                            .optArgument("Content-Type", "application/json")
                            .optArgument("Accept", "*/*")
                            .build();
            try (ZooModel<TestData, TestData> model = criteria2.loadModel();
                    Predictor<TestData, TestData> predictor = model.newPredictor()) {
                TestData ret = predictor.predict(testInput);
                Assert.assertEquals(ret.data, "echo");

                // test SSE response
                server.setContent("data: line1\n\ndata:  line2\n\n");
                server.setContentType("text/event-stream");
                ret = predictor.predict(testInput);
                Assert.assertEquals(ret.data, "line1 line2");

                server.setCode(400);
                Assert.assertThrows(() -> predictor.predict(testInput));
            }
        }
    }

    private static final class TestData implements JsonSerializable {

        private static final long serialVersionUID = 1L;

        String data;

        public TestData(String data) {
            this.data = data;
        }

        public static TestData fromJson(String json) {
            return JsonUtils.GSON.fromJson(json, TestData.class);
        }

        public static TestData fromJson(Iterator<String> it) {
            StringBuilder sb = new StringBuilder();
            it.forEachRemaining(sb::append);
            return new TestData(sb.toString());
        }

        /** {@inheritDoc} */
        @Override
        public JsonElement serialize() {
            return JsonUtils.GSON.toJsonTree(data);
        }
    }
}

/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.translate;

import ai.djl.Model;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.util.JsonUtils;
import ai.djl.util.Pair;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import java.io.BufferedReader;
import java.io.StringReader;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/** A {@link TranslatorFactory} that creates a {@code RawTranslator} instance. */
public class NoopServingTranslatorFactory implements TranslatorFactory {

    /** {@inheritDoc} */
    @Override
    public Set<Pair<Type, Type>> getSupportedTypes() {
        return Collections.singleton(new Pair<>(Input.class, Output.class));
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public <I, O> Translator<I, O> newInstance(
            Class<I> input, Class<O> output, Model model, Map<String, ?> arguments) {
        if (!isSupported(input, output)) {
            throw new IllegalArgumentException("Unsupported input/output types.");
        }
        String batchifier = ArgumentsUtil.stringValue(arguments, "batchifier", "none");
        return (Translator<I, O>) new NoopServingTranslator(Batchifier.fromString(batchifier));
    }

    static final class NoopServingTranslator implements Translator<Input, Output> {

        private Batchifier batchifier;

        NoopServingTranslator(Batchifier batchifier) {
            this.batchifier = batchifier;
        }

        /** {@inheritDoc} */
        @Override
        public Batchifier getBatchifier() {
            return batchifier;
        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, Input input) throws TranslateException {
            NDManager manager = ctx.getNDManager();
            try {
                ctx.setAttachment("properties", input.getProperties());
                String contentType = input.getProperty("Content-Type", null);
                if (contentType != null) {
                    int pos = contentType.indexOf(';');
                    if (pos > 0) {
                        contentType = contentType.substring(0, pos);
                    }
                    if ("application/json".equals(contentType)) {
                        String data = input.getData().getAsString();
                        JsonElement element = JsonUtils.GSON.fromJson(data, JsonElement.class);
                        if (element.isJsonObject()) {
                            JsonObject obj = element.getAsJsonObject();
                            element = obj.get("inputs");
                            if (element == null) {
                                element = obj.get("instance");
                            }
                        }
                        if (element != null && element.isJsonArray()) {
                            return toNDList(manager, element);
                        } else {
                            throw new TranslateException("Input is not a supported json format");
                        }
                    } else if ("text/csv".equals(contentType)) {
                        String data = input.getData().getAsString();
                        return parseCsv(manager, data);
                    }
                }

                return input.getDataAsNDList(manager);
            } catch (Exception e) {
                throw new TranslateException("Input processing failed", e);
            }
        }

        /** {@inheritDoc} */
        @Override
        @SuppressWarnings("unchecked")
        public Output processOutput(TranslatorContext ctx, NDList list) {
            Map<String, String> prop = (Map<String, String>) ctx.getAttachment("properties");
            String contentType = prop.getOrDefault("Content-Type", "tensor/ndlist");
            int pos = contentType.indexOf(';');
            if (pos > 0) {
                contentType = contentType.substring(0, pos);
            }

            String accept = prop.get("Accept");
            Output output = new Output();
            if ("tensor/npz".equalsIgnoreCase(accept)
                    || "tensor/npz".equalsIgnoreCase(contentType)) {
                output.add(list.encode(NDList.Encoding.NPZ));
                output.addProperty("Content-Type", "tensor/npz");
            } else if ("tensor/safetensors".equalsIgnoreCase(accept)
                    || "tensor/safetensors".equalsIgnoreCase(contentType)) {
                output.add(list.encode(NDList.Encoding.SAFETENSORS));
                output.addProperty("Content-Type", "tensor/safetensors");
            } else if ("text/csv".equalsIgnoreCase(accept)) {
                String csv = toCsv(list);
                output.add(
                        BytesSupplier.wrap(csv.getBytes(java.nio.charset.StandardCharsets.UTF_8)));
                output.addProperty("Content-Type", "text/csv");
            } else if ("application/json".equalsIgnoreCase(accept)
                    || "application/json".equalsIgnoreCase(contentType)) {
                List<Object> ret;
                if (list.size() == 1) {
                    ret = toList(list.get(0));
                } else {
                    ret = new ArrayList<>();
                    for (NDArray array : list) {
                        ret.add(new Pair<>(array.getName(), toList(array)));
                    }
                }
                Map<String, List<Object>> map = new ConcurrentHashMap<>();
                map.put("predictions", ret);
                output.add("predictions", BytesSupplier.wrapAsJson(map));

            } else {
                output.add(list.encode());
                output.addProperty("Content-Type", "tensor/ndlist");
            }
            return output;
        }

        private NDList toNDList(NDManager manager, JsonElement element) {
            JsonElement e = element.getAsJsonArray().get(0);
            if (e.isJsonArray()) {
                float[][] array = JsonUtils.GSON.fromJson(element, float[][].class);
                return new NDList(manager.create(array));
            } else {
                float[] array = JsonUtils.GSON.fromJson(element, float[].class);
                return new NDList(manager.create(array));
            }
        }

        private List<Object> toList(NDArray array) {
            Number[] data = array.toArray();
            long[] shape = array.getShape().getShape();
            return toList(Arrays.asList(data).iterator(), shape, 0);
        }

        private List<Object> toList(Iterator<Number> data, long[] shape, int pos) {
            List<Object> ret = new ArrayList<>();
            if (pos == shape.length - 1) {
                for (int i = 0; i < shape[pos]; ++i) {
                    ret.add(data.next());
                }
                return ret;
            }
            for (int i = 0; i < shape[pos]; ++i) {
                ret.add(toList(data, shape, pos + 1));
            }
            return ret;
        }

        private NDList parseCsv(NDManager manager, String csvData) throws TranslateException {
            try (BufferedReader reader = new BufferedReader(new StringReader(csvData))) {
                List<float[]> rows = new ArrayList<>();
                String line;
                boolean firstLine = true;
                while ((line = reader.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty()) {
                        continue;
                    }

                    // Skip header if it contains non-numeric data
                    if (firstLine && !isNumericRow(line)) {
                        firstLine = false;
                        continue;
                    }
                    firstLine = false;

                    String[] values = line.split(",");
                    float[] row = new float[values.length];
                    for (int i = 0; i < values.length; i++) {
                        row[i] = Float.parseFloat(values[i].trim());
                    }
                    rows.add(row);
                }

                if (rows.isEmpty()) {
                    throw new TranslateException("No valid numeric data found in CSV");
                }

                float[][] data = rows.toArray(new float[0][]);
                return new NDList(manager.create(data));
            } catch (IOException | NumberFormatException e) {
                throw new TranslateException("Failed to parse CSV data", e);
            }
        }

        private boolean isNumericRow(String line) {
            String[] values = line.split(",");
            for (String value : values) {
                try {
                    Float.parseFloat(value.trim());
                } catch (NumberFormatException e) {
                    return false;
                }
            }
            return true;
        }

        private String toCsv(NDList list) {
            StringBuilder sb = new StringBuilder();
            for (NDArray array : list) {
                Number[] data = array.toArray();
                long[] shape = array.getShape().getShape();

                if (shape.length == 1) {
                    for (int i = 0; i < data.length; i++) {
                        if (i > 0) {
                            sb.append(',');
                        }
                        sb.append(data[i]);
                    }
                    sb.append('\n');
                } else if (shape.length == 2) {
                    int cols = (int) shape[1];
                    for (int i = 0; i < data.length; i++) {
                        if (i > 0 && i % cols == 0) {
                            sb.append('\n');
                        }
                        if (i % cols > 0) {
                            sb.append(',');
                        }
                        sb.append(data[i]);
                    }
                    sb.append('\n');
                }
            }
            return sb.toString();
        }
    }
}

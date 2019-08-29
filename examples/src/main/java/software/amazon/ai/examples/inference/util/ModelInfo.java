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
package software.amazon.ai.examples.inference.util;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/** A class hold information of model zoo models and their URLs. */
public class ModelInfo {

    public static final int IMAGE_CLASSIFICATION = 1;
    public static final int FACE_RECOGNITION = 2;
    public static final int SEMANTIC_SEGMENTATION = 3;
    public static final int EMOTION_DETECTION = 4;
    public static final int LANGUAGE_MODELING = 5;

    private static final String S3_PREFIX =
            "https://s3.amazonaws.com/model-server/model_archive_1.0/";

    private static final Map<String, ModelInfo> MODELS = new ConcurrentHashMap<>();

    static {
        MODELS.put("FERPlus", new ModelInfo("FERPlus", ModelInfo.EMOTION_DETECTION));
        MODELS.put("caffenet", new ModelInfo("caffenet"));
        MODELS.put("Inception-BN", new ModelInfo("inception-bn"));
        MODELS.put("lstm_ptb", new ModelInfo("lstm_ptb", ModelInfo.LANGUAGE_MODELING));
        MODELS.put("nin", new ModelInfo("nin"));
        MODELS.put(
                "onnx-arcface-resnet100",
                new ModelInfo("onnx-arcface-resnet100", ModelInfo.FACE_RECOGNITION));
        MODELS.put("onnx-duc", new ModelInfo("onnx-duc", ModelInfo.SEMANTIC_SEGMENTATION));
        MODELS.put("inception_v1", new ModelInfo("onnx-inception_v1"));
        MODELS.put("mobilenetv2-1.0", new ModelInfo("onnx-mobilenet"));
        MODELS.put("resnet101v1", new ModelInfo("onnx-resnet101v1"));
        MODELS.put("resnet101v2", new ModelInfo("onnx-resnet101v2"));
        MODELS.put("resnet152v1", new ModelInfo("onnx-resnet152v1"));
        MODELS.put("resnet152v2", new ModelInfo("onnx-resnet152v2"));
        MODELS.put("resnet18v1", new ModelInfo("onnx-resnet18v1"));
        MODELS.put("resnet18v2", new ModelInfo("onnx-resnet18v2"));
        MODELS.put("resnet34v1", new ModelInfo("onnx-resnet34v1"));
        MODELS.put("resnet34v2", new ModelInfo("onnx-resnet34v2"));
        MODELS.put("resnet50v1", new ModelInfo("onnx-resnet50v1"));
        MODELS.put("resnet50v2", new ModelInfo("onnx-resnet50v2"));
        MODELS.put("squeezenet", new ModelInfo("onnx-squeezenet"));
        MODELS.put("onnx-vgg16", new ModelInfo("onnx-vgg16"));
        MODELS.put("vgg16_bn", new ModelInfo("onnx-vgg16_bn"));
        MODELS.put("onnx-vgg19", new ModelInfo("onnx-vgg19"));
        MODELS.put("vgg19_bn", new ModelInfo("onnx-vgg19_bn"));
        MODELS.put("resnet-152", new ModelInfo("resnet-152"));
        MODELS.put("resnet-18", new ModelInfo("resnet-18"));
        MODELS.put("resnet50_ssd_model", new ModelInfo("resnet50_ssd"));
        MODELS.put("resnext-101-64x4d", new ModelInfo("resnext-101-64x4d"));
        MODELS.put("squeezenet_v1.1", new ModelInfo("squeezenet_v1.1"));
        MODELS.put("squeezenet_v1.2", new ModelInfo("squeezenet_v1.2"));
        MODELS.put("vgg16", new ModelInfo("vgg16"));
        MODELS.put("vgg19", new ModelInfo("vgg19"));
    }

    private String modelName;
    private String url;
    private int type;

    public ModelInfo(String modelName) {
        this(modelName, IMAGE_CLASSIFICATION);
    }

    public ModelInfo(String modelName, int type) {
        this.modelName = modelName;
        url = S3_PREFIX + modelName + ".mar";
        this.type = type;
    }

    public ModelInfo(String modelName, String url) {
        this(modelName, url, IMAGE_CLASSIFICATION);
    }

    public ModelInfo(String modelName, String url, int type) {
        this.modelName = modelName;
        this.url = url;
        this.type = type;
    }

    public static ModelInfo getModel(String modelName) {
        return MODELS.get(modelName);
    }

    public String getModelName() {
        return modelName;
    }

    public String getUrl() {
        return url;
    }

    public int getType() {
        return type;
    }

    public void download() throws IOException {
        URL downloadUrl = new URL(url);

        String userHome = System.getProperty("user.home");
        Path modelZooDir = Paths.get(userHome).resolve(".model_zoo");
        Path dest = modelZooDir.resolve(modelName);
        if (Files.exists(dest)) {
            return;
        }

        Path tmp = modelZooDir.resolve("tmp/" + modelName).toAbsolutePath();
        Files.createDirectories(tmp);

        try (InputStream is = downloadUrl.openStream()) {
            unzip(is, tmp);
        }

        Files.move(tmp, dest);
    }

    public Path getDownloadDir() {
        String userHome = System.getProperty("user.home");
        return Paths.get(userHome).resolve(".model_zoo/" + modelName);
    }

    public static void unzip(InputStream is, Path dest) throws IOException {
        try (ZipInputStream zis = new ZipInputStream(is)) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                String name = entry.getName();
                Path file = dest.resolve(name);
                if (entry.isDirectory()) {
                    Files.createDirectories(file);
                } else {
                    Path parentFile = file.getParent();
                    if (parentFile == null) {
                        throw new AssertionError(
                                "Parent path should never be null: " + file.toString());
                    }
                    Files.createDirectories(parentFile);
                    Files.copy(zis, file);
                }
            }
        }
    }
}

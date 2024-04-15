/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.llama.zoo;

import ai.djl.Application;
import ai.djl.engine.Engine;
import ai.djl.repository.Repository;
import ai.djl.repository.Version;
import ai.djl.repository.zoo.ModelLoader;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.util.ClassLoaderUtils;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;

import com.google.gson.reflect.TypeToken;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.io.Reader;
import java.io.Writer;
import java.lang.reflect.Type;
import java.net.URI;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.zip.GZIPInputStream;

/** LlamaModelZoo is a repository that contains llama.cpp models. */
public class LlamaModelZoo extends ModelZoo {

    private static final Logger logger = LoggerFactory.getLogger(LlamaModelZoo.class);

    private static final String REPO = "https://mlrepo.djl.ai/";
    private static final Repository REPOSITORY = Repository.newInstance("gguf", REPO);
    private static final String GROUP_ID = "ai.djl.huggingface.gguf";

    private static final long ONE_DAY = Duration.ofDays(1).toMillis();

    private volatile boolean initialized = false;

    LlamaModelZoo() {}

    /** {@inheritDoc} */
    @Override
    public String getGroupId() {
        return GROUP_ID;
    }

    /** {@inheritDoc} */
    @Override
    public Set<String> getSupportedEngines() {
        return Collections.singleton("Llama");
    }

    /** {@inheritDoc} */
    @Override
    public Collection<ModelLoader> getModelLoaders() {
        init();
        return super.getModelLoaders();
    }

    /** {@inheritDoc} */
    @Override
    public ModelLoader getModelLoader(String name) {
        init();
        return super.getModelLoader(name);
    }

    private void init() {
        if (!initialized) {
            synchronized (LlamaModelZoo.class) {
                if (!initialized) {
                    Application app = Application.NLP.TEXT_GENERATION;
                    Map<String, ModelDetail> map = listModels(app);
                    for (Map.Entry<String, ModelDetail> entry : map.entrySet()) {
                        String artifactId = entry.getKey();
                        Map<String, Object> gguf = entry.getValue().getGguf();
                        if (gguf != null) {
                            for (String key : gguf.keySet()) {
                                addModel(REPOSITORY.model(app, GROUP_ID, artifactId, "0.0.1", key));
                            }
                        }
                    }
                    initialized = true;
                }
            }
        }
    }

    private Map<String, ModelDetail> listModels(Application app) {
        try {
            String path = "model/" + app.getPath() + "/ai/djl/huggingface/gguf/";
            Path dir = Utils.getCacheDir().resolve("cache/repo/" + path);
            if (Files.notExists(dir)) {
                Files.createDirectories(dir);
            } else if (!Files.isDirectory(dir)) {
                logger.warn("Failed initialize cache directory: " + dir);
                return Collections.emptyMap();
            }
            Type type = new TypeToken<Map<String, ModelDetail>>() {}.getType();

            Path file = dir.resolve("models.json");
            if (Files.exists(file)) {
                long lastModified = Files.getLastModifiedTime(file).toMillis();
                if (Utils.isOfflineMode() || System.currentTimeMillis() - lastModified < ONE_DAY) {
                    try (Reader reader = Files.newBufferedReader(file)) {
                        return JsonUtils.GSON.fromJson(reader, type);
                    }
                }
            }

            URL url = URI.create(REPO).resolve(path + "models.json.gz").toURL();
            Path tmp = Files.createTempFile(dir, "models", ".tmp");
            try (GZIPInputStream gis = new GZIPInputStream(Utils.openUrl(url))) {
                String json = Utils.toString(gis);
                try (Writer writer = Files.newBufferedWriter(tmp)) {
                    writer.write(json);
                }
                Utils.moveQuietly(tmp, file);
                return JsonUtils.GSON.fromJson(json, type);
            } catch (IOException e) {
                logger.warn("Failed to download Huggingface gguf index: {}", app);
                if (Files.exists(file)) {
                    try (Reader reader = Files.newBufferedReader(file)) {
                        return JsonUtils.GSON.fromJson(reader, type);
                    }
                }

                String resource = app.getPath() + "/" + GROUP_ID + ".json";
                try (InputStream is = ClassLoaderUtils.getResourceAsStream(resource)) {
                    String json = Utils.toString(is);
                    try (Writer writer = Files.newBufferedWriter(tmp)) {
                        writer.write(json);
                    }
                    Utils.moveQuietly(tmp, file);
                    return JsonUtils.GSON.fromJson(json, type);
                }
            } finally {
                Utils.deleteQuietly(tmp);
            }
        } catch (IOException e) {
            logger.warn("Failed load gguf index file", e);
        }

        return Collections.emptyMap();
    }

    private static final class ModelDetail {

        private Map<String, Object> gguf;

        public Map<String, Object> getGguf() {
            return gguf;
        }

        public void setGguf(Map<String, Object> gguf) {
            this.gguf = gguf;
        }
    }
}

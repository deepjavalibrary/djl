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
package ai.djl.huggingface.zoo;

import ai.djl.Application;
import ai.djl.Application.NLP;
import ai.djl.engine.Engine;
import ai.djl.repository.Repository;
import ai.djl.repository.Version;
import ai.djl.repository.VersionRange;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;

import com.google.gson.reflect.TypeToken;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.lang.reflect.Type;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import java.util.zip.GZIPInputStream;

/** HfModelZoo is a repository that contains HuggingFace models. */
public class HfModelZoo extends ModelZoo {

    private static final Logger logger = LoggerFactory.getLogger(HfModelZoo.class);

    private static final String REPO = "https://mlrepo.djl.ai/";
    private static final Repository REPOSITORY = Repository.newInstance("Huggingface", REPO);
    private static final String GROUP_ID = "ai.djl.huggingface.pytorch";

    private static final long ONE_DAY = Duration.ofDays(1).toMillis();

    HfModelZoo() {
        Version version = new Version(Engine.class.getPackage().getSpecificationVersion());
        addModels(NLP.FILL_MASK, version);
        addModels(NLP.QUESTION_ANSWER, version);
        addModels(NLP.TEXT_CLASSIFICATION, version);
        addModels(NLP.TEXT_EMBEDDING, version);
        addModels(NLP.TOKEN_CLASSIFICATION, version);
    }

    /** {@inheritDoc} */
    @Override
    public String getGroupId() {
        return GROUP_ID;
    }

    /** {@inheritDoc} */
    @Override
    public Set<String> getSupportedEngines() {
        return Collections.singleton("PyTorch");
    }

    private void addModels(Application app, Version version) {
        Map<String, Map<String, Object>> map = listModels(app);
        for (Map.Entry<String, Map<String, Object>> entry : map.entrySet()) {
            Map<String, Object> model = entry.getValue();
            if ("failed".equals(model.get("result"))) {
                continue;
            }
            String requires = (String) model.get("requires");
            if (requires != null) {
                // the model requires specific DJL version
                VersionRange range = VersionRange.parse(requires);
                if (!range.contains(version)) {
                    continue;
                }
            }
            String artifactId = entry.getKey();
            addModel(REPOSITORY.model(app, GROUP_ID, artifactId, "0.0.1"));
        }
    }

    private Map<String, Map<String, Object>> listModels(Application app) {
        try {
            String path = "model/" + app.getPath() + "/ai/djl/huggingface/pytorch/";
            Path dir = Utils.getCacheDir().resolve("cache/repo/" + path);
            if (Files.notExists(dir)) {
                Files.createDirectories(dir);
            } else if (!Files.isDirectory(dir)) {
                logger.warn("Failed initialize cache directory: " + dir);
                return Collections.emptyMap();
            }
            Type type = new TypeToken<Map<String, Map<String, Object>>>() {}.getType();

            Path file = dir.resolve("models.json");
            if (Files.exists(file)) {
                long lastModified = Files.getLastModifiedTime(file).toMillis();
                if (Boolean.getBoolean("offline")
                        || System.currentTimeMillis() - lastModified < ONE_DAY) {
                    try (Reader reader = Files.newBufferedReader(file)) {
                        return JsonUtils.GSON.fromJson(reader, type);
                    }
                }
            }

            String url = REPO + path + "models.json.gz";
            Path tmp = Files.createTempFile(dir, "models", ".tmp");
            try (GZIPInputStream gis = new GZIPInputStream(new URL(url).openStream())) {
                String json = Utils.toString(gis);
                try (Writer writer = Files.newBufferedWriter(tmp)) {
                    writer.write(json);
                }
                Utils.moveQuietly(tmp, file);
                return JsonUtils.GSON.fromJson(json, type);
            } finally {
                Utils.deleteQuietly(tmp);
            }
        } catch (IOException e) {
            logger.warn("Failed load index of models: " + app, e);
        }

        return Collections.emptyMap();
    }
}

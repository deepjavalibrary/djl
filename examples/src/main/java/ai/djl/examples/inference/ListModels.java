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
package ai.djl.examples.inference;

import ai.djl.Application;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;
import java.util.Map;

public final class ListModels {

    private static final Logger logger = LoggerFactory.getLogger(ListModels.class);

    private ListModels() {}

    public static void main(String[] args) throws IOException, ModelNotFoundException {
        boolean withArtifacts =
                args.length > 0 && ("--artifact".equals(args[0]) || "-a".equals(args[0]));
        if (!withArtifacts) {
            logger.info("============================================================");
            logger.info("user ./gradlew listModel --args='-a' to show artifact detail");
            logger.info("============================================================");
        }
        Map<Application, List<MRL>> models = ModelZoo.listModels();
        for (Map.Entry<Application, List<MRL>> entry : models.entrySet()) {
            String appName = entry.getKey().toString();
            for (MRL mrl : entry.getValue()) {
                if (withArtifacts) {
                    for (Artifact artifact : mrl.listArtifacts()) {
                        logger.info("{} djl://{}", appName, artifact);
                    }
                } else {
                    logger.info("{} {}", appName, mrl);
                }
            }
        }
    }
}

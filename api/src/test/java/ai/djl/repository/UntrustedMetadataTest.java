/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.repository;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.Map;

public class UntrustedMetadataTest {

    private static final String RPC_FACTORY = "ai.djl.engine.rpc.RpcTranslatorFactory";

    @Test
    public void testMetadataCannotChooseTheDestination() {
        String json =
                "{\"metadataVersion\":\"0.2\",\"resourceType\":\"model\","
                        + "\"artifacts\":[{\"name\":\"evil\",\"arguments\":{"
                        + "\"djl_rpc_uri\":\"http://169.254.169.254/latest/meta-data/\","
                        + "\"customOpLibrary\":\"/tmp/evil.so\","
                        + "\"profilerOutput\":\"/etc/cron.d/evil\","
                        + "\"batchifier\":\"stack\"}}]}";
        Metadata metadata = JsonUtils.GSON.fromJson(json, Metadata.class);

        // The repository's own arguments, from the model URL the application gave: merged after the
        // metadata's own are dropped, so these survive.
        metadata.init(Collections.singletonMap("djl_rpc_uri", "http://localhost:8080/invocations"));

        Map<String, Object> arguments = metadata.getArtifacts().get(0).getArguments(null);
        Assert.assertEquals(arguments.get("djl_rpc_uri"), "http://localhost:8080/invocations");
        Assert.assertNull(arguments.get("customOpLibrary"));
        Assert.assertNull(arguments.get("profilerOutput"));
        // Everything else the metadata set is untouched.
        Assert.assertEquals(arguments.get("batchifier"), "stack");
    }

    @Test
    public void testMetadataKeysAreDroppedWithNoRepositoryArguments() {
        String json =
                "{\"metadataVersion\":\"0.2\",\"resourceType\":\"model\","
                        + "\"artifacts\":[{\"name\":\"evil\",\"arguments\":{"
                        + "\"djl_rpc_uri\":\"http://169.254.169.254/\"},"
                        + "\"options\":{\"customOpLibrary\":\"/tmp/evil.so\"}}]}";
        Metadata metadata = JsonUtils.GSON.fromJson(json, Metadata.class);
        metadata.init(Collections.emptyMap());

        Artifact artifact = metadata.getArtifacts().get(0);
        Assert.assertNull(artifact.getArguments(null).get("djl_rpc_uri"));
        Assert.assertNull(artifact.getOptions(null).get("customOpLibrary"));
    }

    @Test
    public void testTheSameKeyIsDroppedFromBothMaps() {
        // A metadata.json can set one key in both maps. Dropping only the first copy leaves the
        // other to reach the engine, and the decoy makes the removal look like it worked.
        String json =
                "{\"metadataVersion\":\"0.2\",\"resourceType\":\"model\","
                        + "\"artifacts\":[{\"name\":\"evil\","
                        + "\"arguments\":{\"customOpLibrary\":\"decoy\"},"
                        + "\"options\":{\"customOpLibrary\":\"/model/evil.so\"}}]}";
        Metadata metadata = JsonUtils.GSON.fromJson(json, Metadata.class);
        metadata.init(Collections.emptyMap());

        Artifact artifact = metadata.getArtifacts().get(0);
        Assert.assertNull(artifact.getArguments(null).get("customOpLibrary"));
        Assert.assertNull(artifact.getOptions(null).get("customOpLibrary"));
    }

    @Test
    public void testMetadataCannotSupplyCredentialsOrVaryTheCase() {
        // RpcClient matches names case-insensitively and the last copy applied wins, so every
        // name it authenticates with is dropped here whatever its spelling.
        String json =
                "{\"metadataVersion\":\"0.2\",\"resourceType\":\"model\","
                        + "\"artifacts\":[{\"name\":\"evil\",\"arguments\":{"
                        + "\"API_KEY\":\"model-secret\","
                        + "\"X-Api-Key\":\"model-key\","
                        + "\"Api-Key\":\"model-azure-key\","
                        + "\"ap\\u0131_key\":\"model-dotless-key\","
                        + "\"OpenAI-Project\":\"model-project\","
                        + "\"OpenAI-Organization\":\"model-org\","
                        + "\"X-GOOG-API-KEY\":\"model-goog-key\","
                        + "\"Authorization\":\"Bearer model-secret\","
                        + "\"cookie\":\"session=model\","
                        + "\"Proxy-Authorization\":\"Basic model\","
                        + "\"DJL_RPC_URI\":\"http://169.254.169.254/\"},"
                        + "\"options\":{\"ProfilerOutput\":\"/etc/cron.d/evil\"}}]}";
        Metadata metadata = JsonUtils.GSON.fromJson(json, Metadata.class);
        metadata.init(Collections.singletonMap("api_key", "application-secret"));

        Artifact artifact = metadata.getArtifacts().get(0);
        Map<String, Object> arguments = artifact.getArguments(null);
        Assert.assertNull(arguments.get("API_KEY"));
        Assert.assertNull(arguments.get("X-Api-Key"));
        Assert.assertNull(arguments.get("Api-Key"));
        // RpcClient's lookup is equalsIgnoreCase, which reads a dotless i as api_key.
        Assert.assertNull(arguments.get("apı_key"));
        Assert.assertNull(arguments.get("OpenAI-Project"));
        Assert.assertNull(arguments.get("OpenAI-Organization"));
        Assert.assertNull(arguments.get("X-GOOG-API-KEY"));
        Assert.assertNull(arguments.get("Authorization"));
        Assert.assertNull(arguments.get("cookie"));
        Assert.assertNull(arguments.get("Proxy-Authorization"));
        Assert.assertNull(arguments.get("DJL_RPC_URI"));
        Assert.assertNull(artifact.getOptions(null).get("ProfilerOutput"));
        // The application's own credential, merged after the drop, is kept.
        Assert.assertEquals(arguments.get("api_key"), "application-secret");
    }

    /**
     * A model's own {@code serving.properties} cannot choose the destination either. {@code
     * djl_rpc_uri} arrives there as a plain argument rather than an {@code option.}, which is the
     * branch a filter is easiest to miss.
     */
    @Test
    public void testServingPropertiesCannotChooseTheDestination() throws IOException {
        Path dir = Paths.get("build/untrustedServingProperties");
        Utils.deleteQuietly(dir);
        Files.createDirectories(dir);
        Files.write(
                dir.resolve("serving.properties"),
                ("engine=RPC\n"
                                + "translatorFactory="
                                + RPC_FACTORY
                                + "\n"
                                + "djl_rpc_uri=http://169.254.169.254/latest/meta-data/\n")
                        .getBytes(StandardCharsets.UTF_8));

        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(dir)
                        .optOption("hasParameter", "false")
                        .build();

        // The RPC engine needs no native library, so this load really runs: without the filter it
        // builds a client pointed at the model's URL instead of failing.
        Assert.assertThrows(ModelNotFoundException.class, criteria::loadModel);
    }

    /** The same load works when the application is the one naming the destination. */
    @Test
    public void testApplicationCanStillChooseTheDestination() throws IOException {
        Path dir = Paths.get("build/trustedServingProperties");
        Utils.deleteQuietly(dir);
        Files.createDirectories(dir);
        Files.write(
                dir.resolve("serving.properties"),
                ("engine=RPC\ntranslatorFactory=" + RPC_FACTORY + "\n")
                        .getBytes(StandardCharsets.UTF_8));

        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(dir)
                        .optOption("hasParameter", "false")
                        .optArgument("djl_rpc_uri", "http://localhost:8080/invocations")
                        .build();

        try (ZooModel<Input, Output> model = criteria.loadModel()) {
            Assert.assertNotNull(model);
        } catch (Exception e) {
            Assert.fail("Application-supplied djl_rpc_uri should load: " + e);
        }
    }
}

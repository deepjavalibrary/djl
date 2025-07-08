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
package ai.djl.repository;

import ai.djl.Application;
import ai.djl.engine.rpc.RpcEngine;
import ai.djl.engine.rpc.RpcTranslatorFactory;
import ai.djl.repository.zoo.DefaultModelZoo;
import ai.djl.util.Progress;
import ai.djl.util.Utils;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A {@code RpcRepository} is a {@link Repository} as a remote model server.
 *
 * @see Repository
 */
public class RpcRepository extends AbstractRepository {

    private String artifactId;
    private String modelName;

    RpcRepository(String name, URI uri) {
        super(name, uri);
        modelName = arguments.get("model_name");
        artifactId = arguments.get("artifact_id");
        if (artifactId == null) {
            artifactId = "rpc";
        }
        if (modelName == null) {
            modelName = artifactId;
        }
        arguments.put("translatorFactory", RpcTranslatorFactory.class.getName());
        arguments.put("engine", RpcEngine.ENGINE_NAME);
        arguments.put("djl_rpc_uri", uri.toString());
    }

    /** {@inheritDoc} */
    @Override
    public boolean isRemote() {
        return true;
    }

    /** {@inheritDoc} */
    @Override
    public Metadata locate(MRL mrl) throws IOException {
        return getMetadata();
    }

    /** {@inheritDoc} */
    @Override
    public Artifact resolve(MRL mrl, Map<String, String> filter) throws IOException {
        List<Artifact> artifacts = locate(mrl).getArtifacts();
        if (artifacts.isEmpty()) {
            return null;
        }
        return artifacts.get(0);
    }

    /** {@inheritDoc} */
    @Override
    public List<MRL> getResources() {
        MRL mrl = MRL.undefined(this, DefaultModelZoo.GROUP_ID, artifactId);
        return Collections.singletonList(mrl);
    }

    /** {@inheritDoc} */
    @Override
    protected void download(Path tmp, URI baseUri, Artifact.Item item, Progress progress) {}

    private synchronized Metadata getMetadata() {
        Artifact artifact = new Artifact();
        artifact.setName(modelName);
        artifact.getArguments().putAll(arguments);
        Map<String, Artifact.Item> files = new ConcurrentHashMap<>();
        Artifact.Item item = new Artifact.Item();
        item.setUri(uri.getPath());
        item.setName(""); // avoid creating extra folder
        item.setArtifact(artifact);
        item.setSize(0);
        files.put(artifactId, item);
        artifact.setFiles(files);

        Metadata metadata = new Metadata.MatchAllMetadata();
        metadata.setArtifactId(artifactId);
        metadata.setArtifacts(Collections.singletonList(artifact));
        String hash = Utils.hash(uri.toString());
        MRL mrl = model(Application.UNDEFINED, DefaultModelZoo.GROUP_ID, hash);
        metadata.setRepositoryUri(mrl.toURI());
        return metadata;
    }
}

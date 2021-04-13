/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.central.responseencoder;

import ai.djl.repository.Metadata;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;
import java.lang.reflect.Type;
import java.text.DateFormat;

/**
 * json custom serializer for Artifact.Metadata. custom serializer to avoid cyclic serialisation.
 *
 * @author erik.bamberg@web.de
 */
public class MetaDataSerializer implements JsonSerializer<Metadata> {

    /** {@inheritDoc}} */
    @Override
    public JsonElement serialize(Metadata metadata, Type type, JsonSerializationContext ctx) {
        DateFormat dateFormat = DateFormat.getInstance();
        JsonObject jsonMetadata = new JsonObject();

        jsonMetadata.addProperty("artifactId", metadata.getArtifactId());
        jsonMetadata.addProperty("groupId", metadata.getGroupId());
        jsonMetadata.addProperty("description", metadata.getDescription());
        jsonMetadata.addProperty("website", metadata.getWebsite());
        jsonMetadata.addProperty("resourceType", metadata.getResourceType());
        jsonMetadata.addProperty("lastUpdated", dateFormat.format(metadata.getLastUpdated()));
        JsonObject licenses = new JsonObject();
        metadata.getLicenses()
                .forEach(
                        (k, l) -> {
                            licenses.addProperty(k, l.getName());
                        });
        jsonMetadata.add("licenses", licenses);

        return jsonMetadata;
    }
}

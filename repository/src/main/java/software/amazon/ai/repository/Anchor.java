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
package software.amazon.ai.repository;

public class Anchor {

    private String type;
    private String category;
    private String groupId;
    private String artifactId;
    private String version;

    public Anchor(String type, String category, String groupId, String artifactId, String version) {
        this.type = type;
        this.category = category;
        this.groupId = groupId;
        this.artifactId = artifactId;
        this.version = version;
    }

    public static Anchor parse(String anchor) {
        String[] tokens = anchor.split(":");
        if (tokens.length < 4) {
            throw new IllegalArgumentException("Invalid anchor syntax: " + anchor);
        }

        String version = null;
        if (tokens.length == 5) {
            version = tokens[4];
        }

        return new Anchor(tokens[0], tokens[1], tokens[2], tokens[3], version);
    }

    public String getType() {
        return type;
    }

    public String getCategory() {
        return category;
    }

    public String getGroupId() {
        return groupId;
    }

    public String getArtifactId() {
        return artifactId;
    }

    public String getVersion() {
        return version;
    }

    public String getBaseUri() {
        return type + '/' + category + '/' + groupId.replace('.', '/') + '/' + artifactId;
    }
}

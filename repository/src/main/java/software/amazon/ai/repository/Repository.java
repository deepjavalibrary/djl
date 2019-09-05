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

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

public interface Repository {

    Gson GSON =
            new GsonBuilder()
                    .setDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
                    .setPrettyPrinting()
                    .create();

    static Repository newInstance(String name, String url) {
        URI uri = URI.create(url);
        if (!uri.isAbsolute()) {
            return new LocalRepository(name, Paths.get(url));
        }

        String scheme = uri.getScheme();
        if ("file".equalsIgnoreCase(scheme)) {
            return new LocalRepository(name, Paths.get(uri.getPath()));
        }

        return new RemoteRepository(name, uri);
    }

    String getName();

    URI getBaseUri();

    Metadata locate(MRL mrl) throws IOException;

    Artifact resolve(MRL mrl, String version, Map<String, String> filter) throws IOException;

    InputStream openStream(Artifact.Item item, String path) throws IOException;

    String[] listDirectory(Artifact.Item item, String path) throws IOException;

    void prepare(Artifact artifact) throws IOException;

    Path getCacheDirectory() throws IOException;
}

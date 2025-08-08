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
package ai.djl.engine.rpc;

import ai.djl.inference.streaming.ChunkedBytesSupplier;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.util.Utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.Reader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

/** A client used to connect to remote model server. */
public final class RpcClient {

    private static final Logger logger = LoggerFactory.getLogger(RpcClient.class);
    private static final Set<String> RESERVED_KEYS =
            new HashSet<>(
                    Arrays.asList(
                            "engine",
                            "translatorfactory",
                            "translator",
                            "model_name",
                            "artifact_id",
                            "application",
                            "task",
                            "djl_rpc_uri",
                            "method",
                            "api_key",
                            "content-type"));

    private URL url;
    private String method;
    private Map<CaseInsensitiveKey, String> headers;

    private RpcClient(URL url, String method, Map<CaseInsensitiveKey, String> headers) {
        this.url = url;
        this.method = method;
        this.headers = headers;
    }

    /**
     * Returns a new {@code Client} instance from criteria arguments.
     *
     * @param arguments the criteria arguments.
     * @return a new {@code Client} instance
     * @throws MalformedURLException if url is invalid
     */
    public static RpcClient getClient(Map<String, ?> arguments) throws MalformedURLException {
        String url = arguments.get("djl_rpc_uri").toString();
        String method = getOrDefault(arguments, "method", "POST");
        String apiKey = getOrDefault(arguments, "api_key", null);
        String contentType = getOrDefault(arguments, "content-type", "application/json");
        Map<CaseInsensitiveKey, String> httpHeaders = new ConcurrentHashMap<>();
        for (Map.Entry<String, ?> entry : arguments.entrySet()) {
            String key = entry.getKey();
            String value = entry.getValue().toString().trim();
            if (RESERVED_KEYS.contains(key.toLowerCase(Locale.ROOT))) {
                continue;
            }
            httpHeaders.put(new CaseInsensitiveKey(key), value);
        }
        httpHeaders.put(new CaseInsensitiveKey("Content-Type"), contentType);
        if (url.startsWith("https://generativelanguage.googleapis.com")) {
            if (apiKey == null) {
                apiKey = Utils.getenv("GEMINI_API_KEY");
                if (apiKey == null) {
                    apiKey = Utils.getenv("GOOGLE_API_KEY");
                }
            }
            if (apiKey != null) {
                if (url.endsWith("/openai/chat/completions")) {
                    httpHeaders.put(new CaseInsensitiveKey("Authorization"), "Bearer " + apiKey);
                } else {
                    httpHeaders.put(new CaseInsensitiveKey("x-goog-api-key"), apiKey);
                }
            }
        } else if (url.startsWith("https://api.anthropic.com")) {
            if (apiKey == null) {
                apiKey = Utils.getEnvOrSystemProperty("ANTHROPIC_API_KEY");
            }
            if (apiKey != null) {
                httpHeaders.put(new CaseInsensitiveKey("x-api-key"), apiKey);
            }
        }
        if (apiKey == null) {
            apiKey = Utils.getEnvOrSystemProperty("GENAI_API_KEY");
        }
        if (apiKey != null) {
            httpHeaders.put(new CaseInsensitiveKey("Authorization"), "Bearer " + apiKey);
        }
        return new RpcClient(new URL(url), method, httpHeaders);
    }

    /**
     * Sends request to remote server.
     *
     * @param input the input
     * @return the output
     * @throws IOException if connection failed
     */
    public Output send(Input input) throws IOException {
        if (Utils.isOfflineMode()) {
            throw new IOException("Offline mode is enabled.");
        }
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        boolean isStream = false;
        try {
            conn.setRequestMethod(method);
            if ("POST".equals(method) || "PUT".equals(method)) {
                conn.setDoOutput(true);
            }
            Map<String, String> prop = input.getProperties();
            Map<CaseInsensitiveKey, String> reqHeaders = new ConcurrentHashMap<>(headers);
            for (Map.Entry<String, String> entry : prop.entrySet()) {
                reqHeaders.put(new CaseInsensitiveKey(entry.getKey()), entry.getValue());
            }
            for (Map.Entry<CaseInsensitiveKey, String> header : reqHeaders.entrySet()) {
                conn.addRequestProperty(header.getKey().key, header.getValue());
            }

            conn.connect();
            BytesSupplier content = input.getData();
            if (content != null) {
                try (OutputStream os = conn.getOutputStream()) {
                    os.write(content.getAsBytes());
                }
            }
            int code = conn.getResponseCode();
            Output out = new Output(code, conn.getResponseMessage());
            Map<String, List<String>> respHeaders = conn.getHeaderFields();
            for (Map.Entry<String, List<String>> entry : respHeaders.entrySet()) {
                String key = entry.getKey();
                String value = entry.getValue().get(0);
                if (key != null && value != null) {
                    value = value.toLowerCase(Locale.ROOT);
                    if ("content-type".equalsIgnoreCase(key)
                            && (value.startsWith("text/event-stream")
                                    || value.startsWith("application/jsonlines"))) {
                        isStream = true;
                    }
                    out.addProperty(key, value);
                }
            }
            if (code == 200) {
                if (isStream) {
                    ChunkedBytesSupplier cbs = new ChunkedBytesSupplier();
                    out.add(cbs);
                    CompletableFuture.supplyAsync(() -> handleStream(conn, cbs));
                } else {
                    try (InputStream is = conn.getInputStream()) {
                        out.add(Utils.toByteArray(is));
                    }
                }
            } else {
                try (InputStream is = conn.getErrorStream()) {
                    if (is != null) {
                        String error = Utils.toString(is);
                        out.add(error);
                        logger.warn("Failed to invoke model server: {}", error);
                    } else {
                        logger.warn("Failed to invoke model server, code: {}", code);
                    }
                }
            }
            return out;
        } finally {
            if (!isStream) {
                conn.disconnect();
            }
        }
    }

    private static String getOrDefault(Map<String, ?> arguments, String key, String def) {
        for (Map.Entry<String, ?> entry : arguments.entrySet()) {
            if (entry.getKey().equalsIgnoreCase(key)) {
                return entry.getValue().toString();
            }
        }
        return def;
    }

    private static Void handleStream(HttpURLConnection conn, ChunkedBytesSupplier cbs) {
        BytesSupplier pendingChunk = null;
        try (Reader r = new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8);
                BufferedReader reader = new BufferedReader(r)) {
            String line;
            StringBuilder sb = new StringBuilder();
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("data: ")) {
                    if (sb.length() > 0) {
                        sb.append('\n');
                    }
                    sb.append(line.substring(6));
                } else if (line.startsWith("event: ")) {
                    // ignore
                    continue;
                } else if (!line.isEmpty()) {
                    // jsonlines
                    if (pendingChunk != null) {
                        cbs.appendContent(pendingChunk, false);
                    }
                    pendingChunk = BytesSupplier.wrap(line);
                } else if (sb.length() > 0) {
                    if (pendingChunk != null) {
                        cbs.appendContent(pendingChunk, false);
                    }
                    pendingChunk = BytesSupplier.wrap(sb.toString());
                    sb.setLength(0);
                }
            }
        } catch (IOException e) {
            logger.warn("Failed run inference.", e);
            cbs.appendContent(BytesSupplier.wrap("connection abort exceptionally"), false);
        } finally {
            if (pendingChunk == null) {
                pendingChunk = BytesSupplier.wrap(new byte[0]);
            }
            cbs.appendContent(pendingChunk, true);
            conn.disconnect();
        }
        return null;
    }

    static final class CaseInsensitiveKey {
        String key;

        public CaseInsensitiveKey(String key) {
            this.key = key;
        }

        /** {@inheritDoc} */
        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (!(o instanceof CaseInsensitiveKey)) {
                return false;
            }
            CaseInsensitiveKey header = (CaseInsensitiveKey) o;
            return key.equalsIgnoreCase(header.key);
        }

        /** {@inheritDoc} */
        @Override
        public int hashCode() {
            return Objects.hashCode(key.toLowerCase(Locale.ROOT));
        }
    }
}

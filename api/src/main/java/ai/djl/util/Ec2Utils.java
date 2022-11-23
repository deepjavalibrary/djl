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
package ai.djl.util;

import ai.djl.engine.Engine;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.Proxy;
import java.net.URL;

/** A utility class to retrieve EC2 metadata. */
public final class Ec2Utils {

    private static final Logger logger = LoggerFactory.getLogger(Ec2Utils.class);

    private static final String TOKEN_URL = "http://169.254.169.254/latest/api/token";
    private static final String EC2_METADATA = "http://169.254.169.254/latest/meta-data/";

    private Ec2Utils() {}

    /**
     * Returns the EC2 metadata for specified key.
     *
     * @param key the key to retrieve
     * @return the EC2 metadata for specified key
     */
    public static String readMetadata(String key) {
        HttpURLConnection conn = null;
        try {
            String header = "X-aws-ec2-metadata-token";
            String token = getToken();
            String url = EC2_METADATA + key;
            conn = openConnection(new URL(url), "GET", header, token);
            int statusCode = conn.getResponseCode();
            if (statusCode == HttpURLConnection.HTTP_OK) {
                try (InputStream is = conn.getInputStream()) {
                    return Utils.toString(is);
                }
            } else {
                String reason = conn.getResponseMessage();
                logger.debug("Failed to get EC2 metadata: {} {}", statusCode, reason);
            }
        } catch (IOException ignore) {
            // ignore
        } finally {
            if (conn != null) {
                conn.disconnect();
            }
        }
        return null;
    }

    /**
     * Sends telemetry information.
     *
     * @param engine the default engine name
     */
    public static void callHome(String engine) {
        if (Boolean.getBoolean("offline")
                || Boolean.parseBoolean(Utils.getenv("OPT_OUT_TRACKING"))) {
            return;
        }
        String instanceId = Ec2Utils.readMetadata("instance-id");
        if (instanceId == null) {
            return;
        }
        String region = readMetadata("placement/availability-zone");
        if (region == null || region.length() == 0) {
            return;
        }
        region = region.substring(0, region.length() - 1);
        String url =
                "https://djl-telemetry-"
                        + region
                        + ".s3."
                        + region
                        + ".amazonaws.com/telemetry.txt?instance-id="
                        + instanceId
                        + "&version="
                        + Engine.getDjlVersion()
                        + "&engine="
                        + engine;
        HttpURLConnection conn = null;
        try {
            conn = openConnection(new URL(url), "GET", null, null);
            int statusCode = conn.getResponseCode();
            if (statusCode != HttpURLConnection.HTTP_OK) {
                logger.debug("telemetry: {} {}", statusCode, conn.getResponseMessage());
            } else {
                logger.info(
                        "DJL will collect telemetry to help us better understand our users’ needs,"
                            + " diagnose issues, and deliver additional features. If you would like"
                            + " to learn more or opt-out please go to:"
                            + " https://docs.djl.ai/docs/telemetry.html for more information.");
            }
        } catch (IOException e) {
            logger.debug("Failed call home.");
        } finally {
            if (conn != null) {
                conn.disconnect();
            }
        }
    }

    private static String getToken() {
        HttpURLConnection conn = null;
        try {
            String header = "X-aws-ec2-metadata-token-ttl-seconds";
            conn = openConnection(new URL(TOKEN_URL), "PUT", header, "21600");
            int statusCode = conn.getResponseCode();
            if (statusCode == HttpURLConnection.HTTP_OK) {
                try (InputStream is = conn.getInputStream()) {
                    return Utils.toString(is);
                }
            } else {
                logger.debug("EC2 IMDSv2: {} {}", statusCode, conn.getResponseMessage());
            }
        } catch (IOException ignore) {
            // ignore
        } finally {
            if (conn != null) {
                conn.disconnect();
            }
        }
        return null;
    }

    private static HttpURLConnection openConnection(
            URL url, String method, String header, String value) throws IOException {
        HttpURLConnection conn = (HttpURLConnection) url.openConnection(Proxy.NO_PROXY);
        conn.setConnectTimeout(1000);
        conn.setReadTimeout(1000);
        conn.setRequestMethod(method);
        conn.setDoOutput(true);
        conn.addRequestProperty("Accept", "*/*");
        conn.addRequestProperty("User-Agent", "djl");
        if (value != null) {
            conn.addRequestProperty(header, value);
        }
        conn.setInstanceFollowRedirects(false);
        conn.connect();
        return conn;
    }
}

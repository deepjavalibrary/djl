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
package ai.djl.training.util;

import ai.djl.util.Progress;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.net.URLConnection;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.zip.GZIPInputStream;

/** A utility class downloads the file from specified url. */
public final class DownloadUtils {

    private DownloadUtils() {}

    /**
     * Downloads a file from specified url.
     *
     * @param url the url to download
     * @param output the output location
     * @throws IOException when IO operation fails in downloading
     */
    public static void download(String url, String output) throws IOException {
        download(url, output, null);
    }

    /**
     * Downloads a file from specified url.
     *
     * @param url the url to download
     * @param output the output location
     * @param progress the progress tracker to show download progress
     * @throws IOException when IO operation fails in downloading
     */
    public static void download(String url, String output, Progress progress) throws IOException {
        download(new URL(url.trim()), Paths.get(output.trim()), progress);
    }

    /**
     * Downloads a file from specified url.
     *
     * @param url the url to download
     * @param output the output location
     * @param progress the progress tracker to show download progress
     * @throws IOException when IO operation fails in downloading
     */
    public static void download(URL url, Path output, Progress progress) throws IOException {
        if (Files.exists(output)) {
            return;
        }
        Path dir = output.toAbsolutePath().getParent();
        if (dir != null) {
            Files.createDirectories(dir);
        }
        URLConnection conn = url.openConnection();
        if (progress != null) {
            long contentLength = conn.getContentLengthLong();
            if (contentLength > 0) {
                progress.reset("Downloading", contentLength, output.toFile().getName());
            }
        }
        try (InputStream is = conn.getInputStream()) {
            ProgressInputStream pis = new ProgressInputStream(is, progress);
            String fileName = url.getFile();
            if (fileName.endsWith(".gz")) {
                Files.copy(new GZIPInputStream(pis), output, StandardCopyOption.REPLACE_EXISTING);
            } else {
                Files.copy(pis, output, StandardCopyOption.REPLACE_EXISTING);
            }
        }
    }

    private static final class ProgressInputStream extends InputStream {

        private InputStream is;
        private Progress progress;

        public ProgressInputStream(InputStream is, Progress progress) {
            this.is = is;
            this.progress = progress;
        }

        /** {@inheritDoc} */
        @Override
        public int read() throws IOException {
            int ret = is.read();
            if (progress != null) {
                if (ret >= 0) {
                    progress.increment(1);
                } else {
                    progress.end();
                }
            }
            return ret;
        }

        /** {@inheritDoc} */
        @Override
        public int read(byte[] b, int off, int len) throws IOException {
            int size = is.read(b, off, len);
            if (progress != null) {
                progress.increment(size);
            }
            return size;
        }

        /** {@inheritDoc} */
        @Override
        public void close() throws IOException {
            is.close();
        }
    }
}

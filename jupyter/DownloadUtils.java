import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.net.URLConnection;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.zip.GZIPInputStream;
import ai.djl.util.*;

public class DownloadUtils {

    public static void download(String url, String output) throws IOException {
        download(url, output, null);
    }

    public static void download(String url, String output, Progress progress) throws IOException {
        download(new URL(url.trim()), Paths.get(output.trim()), progress);
    }

    public static void download(URL url, Path output, Progress progress) throws IOException {
        if (Files.exists(output)) {
            return;
        }
        Path dir = output.toAbsolutePath().getParent();
        Files.createDirectories(dir);
        URLConnection conn = url.openConnection();
        if (progress != null) {
            long contentLength = conn.getContentLengthLong();
            if (contentLength > 0) {
                progress.reset(output.toFile().getName(), contentLength);
            }
        }
        try (InputStream is = conn.getInputStream()) {
            ProgressInputStream pis = new ProgressInputStream(is, progress);
            String fileName = url.getFile();
            if (fileName.endsWith(".gz")) {
                Files.copy(new GZIPInputStream(pis), output);
            } else {
                Files.copy(pis, output);
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

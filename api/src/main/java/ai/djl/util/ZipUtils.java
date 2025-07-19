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
package ai.djl.util;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

/** Utilities for working with zip files. */
public final class ZipUtils {

    private ZipUtils() {}

    /**
     * Unzips an input stream to a given path.
     *
     * @param is the input stream to unzip
     * @param dest the path to store the unzipped files
     * @throws IOException for failures to unzip the input stream and create files in the dest path
     */
    public static void unzip(InputStream is, Path dest) throws IOException {
        ValidationInputStream vis = new ValidationInputStream(is);
        ZipInputStream zis = new ZipInputStream(vis);
        ZipEntry entry;
        Set<String> set = new HashSet<>();
        while ((entry = zis.getNextEntry()) != null) {
            String entryName = entry.getName();
            // Remove or augment validateArchiveEntry, perform canonical path check
            set.add(entryName);
            Path file = dest.resolve(entryName).normalize();
            // Zip Slip protection: ensure extraction stays within dest
            if (!file.startsWith(dest.normalize())) {
                throw new IOException("Entry is outside of the target dir: " + entryName);
            }
            if (entry.isDirectory()) {
                Files.createDirectories(file);
            } else {
                Path parentFile = file.getParent();
                if (parentFile == null) {
                    throw new AssertionError("Parent path should never be null: " + file);
                }
                Files.createDirectories(parentFile);
                Files.copy(zis, file, StandardCopyOption.REPLACE_EXISTING);
            }
        }
        try {
            // Validate local files against central directory for CVE-2007-4546 and CVE-2014-2720
            vis.validate(set);
        } catch (IOException e) {
            Utils.deleteQuietly(dest);
            throw e;
        }
    }

    /**
     * Zips an input directory to a given file.
     *
     * @param src the input directory to zip
     * @param dest the path to store the zipped files
     * @param includeFolderName if include the source directory name in the zip entry
     * @throws IOException for failures to zip the input directory
     */
    public static void zip(Path src, Path dest, boolean includeFolderName) throws IOException {
        try (ZipOutputStream zos =
                new ZipOutputStream(new BufferedOutputStream(Files.newOutputStream(dest)))) {
            Path root = includeFolderName ? src.getParent() : src;
            if (root == null) {
                throw new AssertionError("Parent folder should not be null.");
            }
            addToZip(root, src, zos);
        }
    }

    private static void addToZip(Path root, Path file, ZipOutputStream zos) throws IOException {
        Path relative = root.relativize(file);
        String name = relative.toString();
        if (Files.isDirectory(file)) {
            if (!name.isEmpty()) {
                ZipEntry entry = new ZipEntry(name + '/');
                zos.putNextEntry(entry);
            }
            File[] files = file.toFile().listFiles();
            if (files != null) {
                for (File f : files) {
                    addToZip(root, f.toPath(), zos);
                }
            }
        } else if (Files.isRegularFile(file)) {
            if (name.isEmpty()) {
                name = file.toFile().getName();
            }
            ZipEntry entry = new ZipEntry(name);
            zos.putNextEntry(entry);
            Files.copy(file, zos);
        }
    }

    static void validateArchiveEntry(String name, Path destination) throws IOException {
        if (name.contains("..")) {
            throw new IOException("Invalid archive entry, contains traversal elements: " + name);
        }
        Path expectedOutputPath = destination.resolve(name).toAbsolutePath().normalize();
        if (!expectedOutputPath.startsWith(destination.normalize())) {
            throw new IOException(
                    "Bad archive entry "
                            + name
                            + ". Attempted write outside destination "
                            + destination);
        }
    }

    private static final class ValidationInputStream extends FilterInputStream {

        private static final int ZIP64_LOCSIG = 0x07064b50; // "PK\006\007"
        private static final int ZIP64_ENDSIG = 0x06064b50; // "PK\006\006"
        private static final int ENDSIG = 0x06054b50; // "PK\005\006"
        private static final int LOCSIG = 0x04034b50; // "PK\003\004"
        private static final int CENSIG = 0x02014b50; // "PK\001\002"

        private static final int ZIP64_LOCHDR = 20; // ZIP64 end loc header size
        private static final int ENDHDR = 22; // END header size
        private static final int CENHDR = 46; // CEN header size
        private static final int USE_UTF8 = 0x800;

        private byte[] buf;
        private boolean seenCen;
        private long filePosition;

        ValidationInputStream(InputStream in) {
            super(in);
            buf = new byte[512];
        }

        /** {@inheritDoc} */
        @Override
        public int read() throws IOException {
            int read = super.read();
            if (read >= 0 && !seenCen) {
                System.arraycopy(buf, 1, buf, 0, buf.length - 1);
                buf[buf.length - 1] = (byte) read;
            }
            return read;
        }

        /** {@inheritDoc} */
        @Override
        public int read(byte[] b, int off, int len) throws IOException {
            int read = super.read(b, off, len);
            if (read > 0 && !seenCen) {
                if (read < buf.length) {
                    System.arraycopy(buf, read, buf, 0, buf.length - read);
                    System.arraycopy(b, off, buf, buf.length - read, read);
                } else {
                    System.arraycopy(b, off + read - buf.length, buf, 0, buf.length);
                }
                filePosition += read;
            }

            return read;
        }

        void validate(Set<String> set) throws IOException {
            seenCen = true;
            if (filePosition > 0) {
                ByteArrayOutputStream bos = new ByteArrayOutputStream();
                if (filePosition < buf.length) {
                    bos.write(buf, (int) (buf.length - filePosition), (int) filePosition);
                    filePosition = 0;
                } else {
                    bos.write(buf);
                    filePosition -= buf.length;
                }
                byte[] tmp = new byte[512];
                int read;
                while ((read = read(tmp)) != -1) {
                    bos.write(tmp, 0, read);
                }
                bos.close();
                byte[] header = bos.toByteArray();
                List<String> entries = initCEN(header);
                for (String name : entries) {
                    if (!set.remove(name)) {
                        throw new IOException("Malicious zip file, missing file: " + name);
                    }
                }
            }
            if (!set.isEmpty()) {
                throw new IOException("Malicious zip file, found hidden " + set.size() + " files.");
            }
        }

        private End findEND(ByteBuffer bb) throws IOException {
            int remaining = bb.remaining();
            if (bb.remaining() == 0) {
                throw new IOException("Zip file is empty");
            }
            End end = new End();

            // Now scan the block backwards for END header signature
            for (int i = remaining - ENDHDR; i >= 0; --i) {
                if (bb.getInt(i) == ENDSIG) {
                    // Found ENDSIG header
                    end.endpos = i;
                    end.cenlen = bb.getInt(i + 12);
                    end.cenoff = bb.getInt(i + 16);
                    int comlen = bb.getShort(i + 20);
                    if (end.endpos + ENDHDR + comlen != remaining) {
                        // ENDSIG matched, however the size of file comment in it does
                        // not match the real size. One "common" cause for this problem
                        // is some "extra" bytes are padded at the end of the zipfile.
                        // Let's do some extra verification, we don't care about the
                        // performance in this situation.
                        int cenpos = end.endpos - end.cenlen;
                        int locpos = Math.toIntExact(cenpos - end.cenoff);
                        if (cenpos < 0
                                || locpos < 0
                                || bb.getInt(cenpos) != CENSIG
                                || bb.getInt(locpos) != LOCSIG) {
                            continue;
                        }
                    }
                    int cenpos = end.endpos - ZIP64_LOCHDR;
                    if (cenpos < 0 || bb.getInt(cenpos) != ZIP64_LOCSIG) {
                        return end;
                    }
                    long end64pos = bb.getLong(cenpos + 8);
                    int relativePos = Math.toIntExact(end64pos - filePosition);
                    if (relativePos < 0 || bb.getInt(relativePos) != ZIP64_ENDSIG) {
                        return end;
                    }

                    // end64 candidate found,
                    int cenlen64 = Math.toIntExact(bb.getLong(relativePos + 40));
                    long cenoff64 = bb.getLong(relativePos + 48);
                    // double-check
                    if (cenlen64 != end.cenlen && end.cenlen > 0
                            || cenoff64 != end.cenoff && end.cenoff > 0) {
                        return end;
                    }
                    // to use the end64 values
                    end.cenlen = cenlen64;
                    end.cenoff = cenoff64;
                    end.endpos = relativePos;

                    return end;
                }
            }
            throw new IOException("Zip END header not found");
        }

        private List<String> initCEN(byte[] header) throws IOException {
            ByteBuffer bb = ByteBuffer.wrap(header);
            bb.order(ByteOrder.LITTLE_ENDIAN);

            End end = findEND(bb);
            if (end.endpos == 0) {
                return Collections.emptyList();
            }

            List<String> entries = new ArrayList<>();

            int cenpos = end.endpos - end.cenlen; // position of CEN table
            int pos = 0;
            while (pos + CENHDR <= end.cenlen) {
                if (bb.getInt(cenpos + pos) != CENSIG) {
                    throw new IOException("invalid CEN header (bad signature)");
                }
                int nlen = bb.getShort(cenpos + pos + 28);
                int elen = bb.getShort(cenpos + pos + 30);
                int clen = bb.getShort(cenpos + pos + 32);
                int flag = bb.getShort(cenpos + pos + 8);
                if ((flag & 1) != 0) {
                    throw new IOException("invalid CEN header (encrypted entry)");
                }
                Charset charset;
                if ((flag & USE_UTF8) != 0) {
                    charset = StandardCharsets.UTF_8;
                } else {
                    charset = StandardCharsets.US_ASCII;
                }
                entries.add(new String(header, cenpos + pos + CENHDR, nlen, charset));

                // skip ext and comment
                pos += (CENHDR + nlen + elen + clen);
            }
            if (pos != end.cenlen) {
                throw new IOException("invalid CEN header (bad header size)");
            }
            return entries;
        }

        private static final class End {
            int cenlen; // 4 bytes
            long cenoff; // 4 bytes
            int endpos; // 4 bytes
        }
    }
}

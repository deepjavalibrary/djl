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
package ai.djl.serving.plugins.staticfile;

import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.util.AsciiString;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** A utility class that handling MIME types. */
public final class MimeUtils {

    private static final Map<String, String> MIME_TYPE_MAP = new ConcurrentHashMap<>();

    static {
        MIME_TYPE_MAP.put("htm", "text/html");
        MIME_TYPE_MAP.put("html", "text/html");
        MIME_TYPE_MAP.put("js", "application/javascript");
        MIME_TYPE_MAP.put("xml", "application/xml");
        MIME_TYPE_MAP.put("css", "text/css");
        MIME_TYPE_MAP.put("txt", "text/plain");
        MIME_TYPE_MAP.put("text", "text/plain");
        MIME_TYPE_MAP.put("csv", "text/comma-separated-values");
        MIME_TYPE_MAP.put("rtf", "text/rtf");
        MIME_TYPE_MAP.put("sh", "text/x-sh");
        MIME_TYPE_MAP.put("tex", "application/x-tex");
        MIME_TYPE_MAP.put("texi", "application/x-texinfo");
        MIME_TYPE_MAP.put("texinfo", "application/x-texinfo");
        MIME_TYPE_MAP.put("t", "application/x-troff");
        MIME_TYPE_MAP.put("tr", "application/x-troff");
        MIME_TYPE_MAP.put("roff", "application/x-troff");
        MIME_TYPE_MAP.put("gif", "image/gif");
        MIME_TYPE_MAP.put("png", "image/x-png");
        MIME_TYPE_MAP.put("ief", "image/ief");
        MIME_TYPE_MAP.put("jpeg", "image/jpeg");
        MIME_TYPE_MAP.put("jpg", "image/jpeg");
        MIME_TYPE_MAP.put("jpe", "image/jpeg");
        MIME_TYPE_MAP.put("tiff", "image/tiff");
        MIME_TYPE_MAP.put("tif", "image/tiff");
        MIME_TYPE_MAP.put("xwd", "image/x-xwindowdump");
        MIME_TYPE_MAP.put("pict", "image/x-pict");
        MIME_TYPE_MAP.put("bmp", "image/x-ms-bmp");
        MIME_TYPE_MAP.put("pcd", "image/x-photo-cd");
        MIME_TYPE_MAP.put("dwg", "image/vnd.dwg");
        MIME_TYPE_MAP.put("dxf", "image/vnd.dxf");
        MIME_TYPE_MAP.put("svf", "image/vnd.svf");
        MIME_TYPE_MAP.put("au", "autio/basic");
        MIME_TYPE_MAP.put("snd", "autio/basic");
        MIME_TYPE_MAP.put("mid", "autio/midi");
        MIME_TYPE_MAP.put("midi", "autio/midi");
        MIME_TYPE_MAP.put("aif", "autio/x-aiff");
        MIME_TYPE_MAP.put("aiff", "autio/x-aiff");
        MIME_TYPE_MAP.put("aifc", "autio/x-aiff");
        MIME_TYPE_MAP.put("wav", "autio/x-wav");
        MIME_TYPE_MAP.put("mpa", "autio/x-mpeg");
        MIME_TYPE_MAP.put("abs", "autio/x-mpeg");
        MIME_TYPE_MAP.put("mpega", "autio/x-mpeg");
        MIME_TYPE_MAP.put("mp2a", "autio/x-mpeg-2");
        MIME_TYPE_MAP.put("mpa2", "autio/x-mpeg-2");
        MIME_TYPE_MAP.put("ra", "application/x-pn-realaudio");
        MIME_TYPE_MAP.put("ram", "application/x-pn-realaudio");
        MIME_TYPE_MAP.put("mpeg", "video/mpeg");
        MIME_TYPE_MAP.put("mpg", "video/mpeg");
        MIME_TYPE_MAP.put("mpe", "video/mpeg");
        MIME_TYPE_MAP.put("mpv2", "video/mpeg-2");
        MIME_TYPE_MAP.put("mp2v", "video/mpeg-2");
        MIME_TYPE_MAP.put("qt", "video/quicktime");
        MIME_TYPE_MAP.put("mov", "video/quicktime");
        MIME_TYPE_MAP.put("avi", "video/x-msvideo");
        MIME_TYPE_MAP.put("ai", "application/postscript");
        MIME_TYPE_MAP.put("eps", "application/postscript");
        MIME_TYPE_MAP.put("ps", "application/postscript");
        MIME_TYPE_MAP.put("pdf", "application/pdf");
        MIME_TYPE_MAP.put("gtar", "application/x-gtar");
        MIME_TYPE_MAP.put("tar", "application/x-tar");
        MIME_TYPE_MAP.put("bcpio", "application/x-bcpio");
        MIME_TYPE_MAP.put("cpio", "application/x-cpio");
        MIME_TYPE_MAP.put("zip", "application/zip");
        MIME_TYPE_MAP.put("rar", "application/rar");
    }

    private MimeUtils() {}

    /**
     * Return the content type that associated with the file.
     *
     * @param fileType file extension
     * @return the content type
     */
    public static AsciiString getContentType(String fileType) {
        String contentType = MIME_TYPE_MAP.get(fileType.toLowerCase(Locale.ROOT));
        if (contentType == null) {
            return HttpHeaderValues.APPLICATION_OCTET_STREAM;
        }
        return AsciiString.cached(contentType);
    }
}

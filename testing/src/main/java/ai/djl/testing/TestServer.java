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
package ai.djl.testing;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;

/** A test model server that can accept inference requests. */
@SuppressWarnings("PMD")
public final class TestServer implements Runnable, AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(TestServer.class);

    private ServerSocket serverSocket;
    private Thread thread;
    private boolean stopped;
    private int code = 200;
    private String content;
    private String contentType;

    private TestServer(ServerSocket serverSocket, String content) {
        this.serverSocket = serverSocket;
        this.content = content;
        this.contentType = "application/json";
    }

    /**
     * Returns a new {@code TestServer} instance.
     *
     * @return a new {@code TestServer} instance
     * @throws IOException if failed to start the listener
     */
    public static TestServer newInstance() throws IOException {
        return newInstance("");
    }

    /**
     * Returns a new {@code TestServer} instance.
     *
     * @param content the server response
     * @return a new {@code TestServer} instance
     * @throws IOException if failed to start the listener
     */
    public static TestServer newInstance(String content) throws IOException {
        ServerSocket serverSocket = new ServerSocket(0);
        TestServer server = new TestServer(serverSocket, content);
        server.thread = new Thread(server);
        server.thread.start();
        return server;
    }

    /**
     * Sets the server response.
     *
     * @param content the server response
     */
    public void setContent(String content) {
        this.content = content;
    }

    /**
     * Sets the server response contentType.
     *
     * @param contentType the server response contentType
     */
    public void setContentType(String contentType) {
        this.contentType = contentType;
    }

    /**
     * Sets the server response code.
     *
     * @param code the server response code
     */
    public void setCode(int code) {
        this.code = code;
    }

    /**
     * Returns the listening port.
     *
     * @return the listening port
     */
    public int getPort() {
        return serverSocket.getLocalPort();
    }

    /** {@inheritDoc} */
    @Override
    public void run() {
        try {
            while (!stopped) {
                try (Socket socket = serverSocket.accept();
                        OutputStream out = socket.getOutputStream();
                        InputStream in = socket.getInputStream()) {
                    readLine(in);
                    readLine(in);
                    int length = 0;
                    while (true) {
                        String line = readLine(in);
                        if (line.isEmpty()) {
                            in.readNBytes(length);
                            break;
                        } else {
                            String[] header = line.split(":");
                            if ("Content-Length".equalsIgnoreCase(header[0].trim())) {
                                length = Integer.parseInt(header[1].trim());
                            }
                        }
                    }
                    out.write(
                            ("HTTP/1.1 "
                                            + code
                                            + " OK\r\nContent-Type: "
                                            + contentType
                                            + "\r\n\r\n")
                                    .getBytes(StandardCharsets.UTF_8));
                    out.write(content.getBytes(StandardCharsets.UTF_8));
                    out.flush();
                } catch (IOException e) {
                    logger.warn("", e);
                }
            }
            serverSocket.close();
        } catch (IOException e) {
            logger.warn("", e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        stopped = true;
        thread.interrupt();
    }

    private String readLine(InputStream is) throws IOException {
        StringBuilder sb = new StringBuilder();
        int c;
        do {
            c = is.read();
            if (c == '\n') {
                break;
            }
            if (c != '\r') {
                sb.append((char) c);
            }
        } while (c != -1);
        return sb.toString();
    }
}

/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.util;

import ai.djl.util.Utils;
import io.netty.channel.Channel;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.ServerChannel;
import io.netty.channel.epoll.Epoll;
import io.netty.channel.epoll.EpollDomainSocketChannel;
import io.netty.channel.epoll.EpollEventLoopGroup;
import io.netty.channel.epoll.EpollServerDomainSocketChannel;
import io.netty.channel.epoll.EpollServerSocketChannel;
import io.netty.channel.epoll.EpollSocketChannel;
import io.netty.channel.kqueue.KQueue;
import io.netty.channel.kqueue.KQueueDomainSocketChannel;
import io.netty.channel.kqueue.KQueueEventLoopGroup;
import io.netty.channel.kqueue.KQueueServerDomainSocketChannel;
import io.netty.channel.kqueue.KQueueServerSocketChannel;
import io.netty.channel.kqueue.KQueueSocketChannel;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.channel.unix.DomainSocketAddress;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.nio.file.Paths;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** A class represents model server's socket listener. */
public final class Connector {

    private static final Pattern ADDRESS_PATTERN =
            Pattern.compile(
                    "((https|http)://([^:^/]+)(:([0-9]+))?)|(unix:(/.*))",
                    Pattern.CASE_INSENSITIVE);

    private static boolean useNativeIo = ConfigManager.getInstance().useNativeIo();

    private boolean uds;
    private String socketPath;
    private String bindIp;
    private int port;
    private boolean ssl;
    private ConnectorType type;

    private Connector(
            int port,
            boolean uds,
            String bindIp,
            String socketPath,
            boolean ssl,
            ConnectorType type) {
        this.port = port;
        this.uds = uds;
        this.bindIp = bindIp;
        this.socketPath = socketPath;
        this.ssl = ssl;
        this.type = type;
    }

    /**
     * Create a {@code Connector} instance based on binding string.
     *
     * @param binding the binding string
     * @param connectorType the type of the connector
     * @return a {@code Connector} instance
     */
    public static Connector parse(String binding, ConnectorType connectorType) {
        Matcher matcher = ADDRESS_PATTERN.matcher(binding);
        if (!matcher.matches()) {
            throw new IllegalArgumentException("Invalid binding address: " + binding);
        }

        boolean uds = matcher.group(7) != null;
        if (uds) {
            if (!useNativeIo) {
                throw new IllegalArgumentException(
                        "unix domain socket requires use_native_io set to true.");
            }
            String path = matcher.group(7);
            return new Connector(-1, true, "", path, false, connectorType);
        }

        String protocol = matcher.group(2);
        String host = matcher.group(3);
        String listeningPort = matcher.group(5);

        boolean ssl = "https".equalsIgnoreCase(protocol);
        int port;
        if (listeningPort == null) {
            port = ssl ? 443 : 80;
        } else {
            port = Integer.parseInt(listeningPort);
        }
        if (port >= 65535) {
            throw new IllegalArgumentException("Invalid port number: " + binding);
        }
        return new Connector(port, false, host, String.valueOf(port), ssl, connectorType);
    }

    /**
     * Returns the socket type.
     *
     * @return the socket type
     */
    public String getSocketType() {
        return uds ? "unix" : "tcp";
    }

    /**
     * Returns if the connector is using unix domain socket.
     *
     * @return {@code true} if the connector is using unix domain socket
     */
    public boolean isUds() {
        return uds;
    }

    /**
     * Return if the connector requires SSL.
     *
     * @return {@code true} if the connector requires SSL
     */
    public boolean isSsl() {
        return ssl;
    }

    /**
     * Returns the unix domain socket path.
     *
     * @return the unix domain socket path
     */
    public String getSocketPath() {
        return socketPath;
    }

    /**
     * Returns the TCP socket listening address.
     *
     * @return the TCP socket listening address
     */
    public SocketAddress getSocketAddress() {
        return uds ? new DomainSocketAddress(socketPath) : new InetSocketAddress(bindIp, port);
    }

    /**
     * Returns the type of the connector.
     *
     * @return the type of the connector
     */
    public ConnectorType getType() {
        return type;
    }

    /**
     * Creates a new netty {@code EventLoopGroup}.
     *
     * @param threads the number of threads
     * @return a new netty {@code EventLoopGroup}
     */
    public static EventLoopGroup newEventLoopGroup(int threads) {
        if (useNativeIo && Epoll.isAvailable()) {
            return new EpollEventLoopGroup(threads);
        } else if (useNativeIo && KQueue.isAvailable()) {
            return new KQueueEventLoopGroup(threads);
        }

        NioEventLoopGroup eventLoopGroup = new NioEventLoopGroup(threads);
        eventLoopGroup.setIoRatio(ConfigManager.getInstance().getIoRatio());
        return eventLoopGroup;
    }

    /**
     * Returns the server channel class.
     *
     * @return the server channel class
     */
    public Class<? extends ServerChannel> getServerChannel() {
        if (useNativeIo && Epoll.isAvailable()) {
            return uds ? EpollServerDomainSocketChannel.class : EpollServerSocketChannel.class;
        } else if (useNativeIo && KQueue.isAvailable()) {
            return uds ? KQueueServerDomainSocketChannel.class : KQueueServerSocketChannel.class;
        }

        return NioServerSocketChannel.class;
    }

    /**
     * Returns the client channel class.
     *
     * @return the client channel class
     */
    public Class<? extends Channel> getClientChannel() {
        if (useNativeIo && Epoll.isAvailable()) {
            return uds ? EpollDomainSocketChannel.class : EpollSocketChannel.class;
        } else if (useNativeIo && KQueue.isAvailable()) {
            return uds ? KQueueDomainSocketChannel.class : KQueueSocketChannel.class;
        }

        return NioSocketChannel.class;
    }

    /** Cleans up the left over resources. */
    public void clean() {
        if (uds) {
            Utils.deleteQuietly(Paths.get(socketPath));
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Connector connector = (Connector) o;
        return uds == connector.uds
                && port == connector.port
                && socketPath.equals(connector.socketPath)
                && bindIp.equals(connector.bindIp);
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return Objects.hash(uds, socketPath, bindIp, port);
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        if (uds) {
            return "unix:" + socketPath;
        } else if (ssl) {
            return "https://" + bindIp + ':' + port;
        }
        return "http://" + bindIp + ':' + port;
    }

    /** An enum represents type of connector. */
    public enum ConnectorType {
        INFERENCE,
        MANAGEMENT,
        BOTH
    }
}

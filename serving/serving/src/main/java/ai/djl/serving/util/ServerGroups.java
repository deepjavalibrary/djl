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

import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.group.ChannelGroup;
import io.netty.channel.group.ChannelGroupFuture;
import io.netty.channel.group.DefaultChannelGroup;
import io.netty.util.concurrent.GlobalEventExecutor;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A class that holds all opened {@code ChannelGroup} and {@code EventLoopGroup}. */
public class ServerGroups {

    static final Logger logger = LoggerFactory.getLogger(ServerGroups.class);

    private ChannelGroup allChannels;

    private EventLoopGroup serverGroup;
    private EventLoopGroup childGroup;

    private ConfigManager configManager;

    /**
     * Constructs a new {@code ServerGroups} instance.
     *
     * @param configManager the configuration
     */
    public ServerGroups(ConfigManager configManager) {
        this.configManager = configManager;
        reset();
    }

    /** Resets the {@code ServerGroups}. */
    public final void reset() {
        allChannels = new DefaultChannelGroup(GlobalEventExecutor.INSTANCE);

        serverGroup = Connector.newEventLoopGroup(2);
        childGroup = Connector.newEventLoopGroup(configManager.getNettyThreads());
    }

    /**
     * Shutdowns all opened channels and event loops.
     *
     * @param graceful {@code ture} for gracefully clean up the resources
     */
    public void shutdown(boolean graceful) {
        closeAllChannels(graceful);

        List<EventLoopGroup> allEventLoopGroups = new ArrayList<>();

        allEventLoopGroups.add(serverGroup);
        allEventLoopGroups.add(childGroup);

        for (EventLoopGroup group : allEventLoopGroups) {
            if (graceful) {
                group.shutdownGracefully();
            } else {
                group.shutdownGracefully(0, 0, TimeUnit.SECONDS);
            }
        }

        if (graceful) {
            for (EventLoopGroup group : allEventLoopGroups) {
                try {
                    group.awaitTermination(60, TimeUnit.SECONDS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        }
    }

    /**
     * Returns the server event loop group.
     *
     * @return the server event loop group
     */
    public EventLoopGroup getServerGroup() {
        return serverGroup;
    }

    /**
     * Returns the client event loop group.
     *
     * @return the client event loop group
     */
    public EventLoopGroup getChildGroup() {
        return childGroup;
    }

    /**
     * Registers a channel to be tracked.
     *
     * @param channel a channel to be tracked
     */
    public void registerChannel(Channel channel) {
        allChannels.add(channel);
    }

    private void closeAllChannels(boolean graceful) {
        ChannelGroupFuture future = allChannels.close();

        // if this is a graceful shutdown, log any channel closing failures. if this isn't a
        // graceful shutdown, ignore them.
        if (graceful) {
            try {
                future.await(10, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

            if (!future.isSuccess()) {
                for (ChannelFuture cf : future) {
                    if (!cf.isSuccess()) {
                        logger.info("Unable to close channel: " + cf.channel(), cf.cause());
                    }
                }
            }
        }
    }
}

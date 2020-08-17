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
package ai.djl.serving;

import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.Connector;
import io.netty.channel.local.LocalChannel;
import org.apache.commons.cli.ParseException;
import org.testng.annotations.Test;

public class ServerInitializerTest {

    @Test
    public void testServerInitializer() throws ParseException {
        ConfigManager.init(ConfigManagerTest.parseArguments(new String[0]));

        ServerInitializer initializer =
                new ServerInitializer(null, Connector.ConnectorType.INFERENCE);
        initializer.initChannel(new LocalChannel());

        initializer = new ServerInitializer(null, Connector.ConnectorType.MANAGEMENT);
        initializer.initChannel(new LocalChannel());
    }
}

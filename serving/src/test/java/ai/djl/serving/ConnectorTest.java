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
import org.apache.commons.cli.ParseException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class ConnectorTest {

    @Test
    public void testConnector() throws ParseException {
        ConfigManager.init(ConfigManagerTest.parseArguments(new String[0]));

        Connector connector =
                Connector.parse("unix:/tmp/mytest", Connector.ConnectorType.INFERENCE);
        Assert.assertEquals(connector.getSocketType(), "unix");
        Assert.assertTrue(connector.isUds());
        Assert.assertEquals(connector.getSocketPath(), "/tmp/mytest");
        Assert.assertEquals(connector.toString(), "unix:/tmp/mytest");

        connector = Connector.parse("http://localhost", Connector.ConnectorType.INFERENCE);
        Assert.assertEquals(connector.toString(), "http://localhost:80");
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testInvalidPort() throws ParseException {
        ConfigManager.init(ConfigManagerTest.parseArguments(new String[0]));

        Connector.parse("http://localhost:65536", Connector.ConnectorType.INFERENCE);
    }
}

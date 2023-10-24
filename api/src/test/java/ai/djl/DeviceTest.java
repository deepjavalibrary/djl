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

package ai.djl;

import ai.djl.Device.MultiDevice;
import ai.djl.engine.Engine;

import org.testng.Assert;
import org.testng.annotations.Test;

public class DeviceTest {

    @Test
    public void testDevice() {
        Assert.assertEquals(Device.cpu(), Device.of("cpu", -1));
        Engine engine = Engine.getInstance();
        if (engine.getGpuCount() > 0) {
            Assert.assertEquals(Device.gpu(), engine.defaultDevice());
        } else {
            Assert.assertEquals(Device.cpu(), engine.defaultDevice());
        }
        Assert.assertEquals(Device.gpu(), Device.of("gpu", 0));
        Assert.assertEquals(Device.gpu(3), Device.of("gpu", 3));
        Assert.assertNotEquals(Device.cpu(), Device.gpu());
        Device dev = Device.of("myDevice", 1);
        Assert.assertEquals(dev.getDeviceType(), "myDevice");

        System.setProperty("test_key", "test");
        Engine.debugEnvironment();

        Assert.assertEquals(2, new MultiDevice(Device.gpu(1), Device.gpu(2)).getDevices().size());
    }

    @Test
    public void testDeviceName() {
        Assert.assertEquals(Device.fromName("cpu"), Device.cpu());
        Assert.assertEquals(Device.fromName("-1"), Device.cpu());

        Assert.assertEquals(Device.fromName("gpu0"), Device.gpu());
        Assert.assertEquals(Device.fromName("0"), Device.gpu());
        Assert.assertEquals(Device.fromName("1"), Device.gpu(1));

        Assert.assertEquals(Device.fromName("nc1"), Device.of("nc", 1));
        Assert.assertEquals(Device.fromName("a999"), Device.of("a", 999));

        Device defaultDevice = Engine.getInstance().defaultDevice();
        Assert.assertEquals(Device.fromName(""), defaultDevice);
        Assert.assertEquals(Device.fromName(null), defaultDevice);

        Assert.assertEquals(
                Device.fromName("gpu1+gpu2"), new MultiDevice(Device.gpu(2), Device.gpu(1)));
        Assert.assertEquals(Device.fromName("gpu1+gpu2"), new MultiDevice("gpu", 1, 3));
    }
}

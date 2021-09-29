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

        Engine.debugEnvironment();
    }
}

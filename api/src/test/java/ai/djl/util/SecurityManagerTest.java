/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import org.testng.Assert;
import org.testng.annotations.AfterTest;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

import java.security.Permission;

public class SecurityManagerTest {

    private SecurityManager originalSM;

    @BeforeTest
    public void setUp() {
        originalSM = System.getSecurityManager();
    }

    @AfterTest
    public void tearDown() {
        System.setSecurityManager(originalSM);
    }

    @Test
    public void testGetenv() {
        // Disable access to system environment
        SecurityManager sm =
                new SecurityManager() {
                    @Override
                    public void checkPermission(Permission perm) {
                        if (perm instanceof RuntimePermission
                                && perm.getName().startsWith("getenv.")) {
                            throw new SecurityException(
                                    "Don't have permission to read system environment: "
                                            + perm.getName());
                        }
                    }
                };
        System.setSecurityManager(sm);

        Assert.assertNull(Utils.getenv("HOME"));
        Assert.assertNull(Utils.getenv("TEST"));
        Assert.assertEquals(Utils.getenv("HOME", "/home"), "/home");
        Assert.assertEquals(Utils.getenv("TEST", "test"), "test");
        Assert.assertEquals(Utils.getenv().size(), 0);
    }
}

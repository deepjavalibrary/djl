/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.google.cloud;

import ai.djl.repository.Repository;

import com.google.cloud.storage.Storage;

import org.mockito.Mockito;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.net.URI;
import java.util.Set;

public class GcsRepositoryFactoryTest {

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testUnsupportedProtocol() {
        GcsRepositoryFactory factory = new GcsRepositoryFactory();
        factory.newInstance("gs", URI.create("https://djl-not-exists/"));
    }

    @Test
    public void testSupportedScheme() {
        GcsRepositoryFactory factory = new GcsRepositoryFactory();
        Set<String> schemes = factory.getSupportedScheme();
        Assert.assertEquals(schemes.size(), 1);
        Assert.assertTrue(schemes.contains("gs"));
    }

    @Test
    public void testGcsRepositoryFactory() {
        Storage storage = Mockito.mock(Storage.class);
        Repository.registerRepositoryFactory(new GcsRepositoryFactory(storage));

        Assert.assertNotNull(
                Repository.newInstance("gs", "gs://djl-not-exists?artifact_id=mlp&model_name=mlp"));
        Assert.assertNotNull(Repository.newInstance("gs", "gs://djl-not-exists"));
        Assert.assertNotNull(Repository.newInstance("gs", "gs://djl-not-exists/?model_name=mlp"));
    }
}

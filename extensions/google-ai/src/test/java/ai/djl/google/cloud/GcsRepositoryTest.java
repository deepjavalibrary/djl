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

import ai.djl.Application;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;

import com.google.api.gax.paging.Page;
import com.google.cloud.storage.Blob;
import com.google.cloud.storage.Storage;

import org.mockito.Mockito;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class GcsRepositoryTest {

    @Test
    public void testGcsRepository() throws IOException {
        Storage storage =
                mockStorage(blob("models/mlp/mlp.pt", 100L), blob("models/mlp/synset.txt", 20L));
        Repository.registerRepositoryFactory(new GcsRepositoryFactory(storage));

        Repository repository = Repository.newInstance("gs", "gs://djl-test/models/mlp");
        Assert.assertTrue(repository.isRemote());

        MRL mrl = repository.model(Application.UNDEFINED, "ai.djl.localmodelzoo", "mlp");
        Artifact artifact = repository.resolve(mrl, null);
        Assert.assertNotNull(artifact);
        Assert.assertEquals(artifact.getName(), "mlp");
        Assert.assertEquals(artifact.getFiles().size(), 2);

        List<MRL> resources = repository.getResources();
        Assert.assertEquals(resources.size(), 1);
    }

    @Test
    public void testEmptyBucket() throws IOException {
        Storage storage = mockStorage();
        Repository.registerRepositoryFactory(new GcsRepositoryFactory(storage));

        Repository repository = Repository.newInstance("gs", "gs://djl-test/non-exists");
        MRL mrl = repository.model(Application.UNDEFINED, "ai.djl.localmodelzoo", "mlp");
        Artifact artifact = repository.resolve(mrl, null);
        Assert.assertNull(artifact);

        List<MRL> resources = repository.getResources();
        Assert.assertTrue(resources.isEmpty());
    }

    @Test
    public void testArchivePrefix() throws IOException {
        Storage storage = mockStorage(blob("models/resnet18.zip", 1000L));
        Repository.registerRepositoryFactory(new GcsRepositoryFactory(storage));

        Repository repository =
                Repository.newInstance(
                        "gs", "gs://djl-test/models/resnet18.zip?model_name=resnet18");
        MRL mrl = repository.model(Application.UNDEFINED, "ai.djl.localmodelzoo", "resnet18");
        Artifact artifact = repository.resolve(mrl, null);
        Assert.assertNotNull(artifact);
        Assert.assertEquals(artifact.getName(), "resnet18");
    }

    private Blob blob(String name, long size) {
        Blob blob = Mockito.mock(Blob.class);
        Mockito.when(blob.getName()).thenReturn(name);
        Mockito.when(blob.getSize()).thenReturn(size);
        return blob;
    }

    @SuppressWarnings("unchecked")
    private Storage mockStorage(Blob... blobs) {
        Storage storage = Mockito.mock(Storage.class);
        Page<Blob> page = Mockito.mock(Page.class);
        Mockito.when(page.getValues())
                .thenReturn(blobs.length == 0 ? Collections.emptyList() : Arrays.asList(blobs));
        Mockito.when(storage.list(Mockito.anyString(), Mockito.any(Storage.BlobListOption[].class)))
                .thenReturn(page);
        return storage;
    }
}

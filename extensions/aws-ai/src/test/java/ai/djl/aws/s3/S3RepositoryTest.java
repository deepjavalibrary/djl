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
package ai.djl.aws.s3;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDList;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import java.io.IOException;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;
import software.amazon.awssdk.auth.credentials.AnonymousCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;

public class S3RepositoryTest {

    @BeforeClass
    public void setUp() {
        System.setProperty("aws.region", "us-east-1");
    }

    @Test
    public void testLoadModelFromS3()
            throws MalformedModelException, ModelNotFoundException, IOException {
        S3Client client =
                S3Client.builder()
                        .credentialsProvider(AnonymousCredentialsProvider.create())
                        .region(Region.US_EAST_1)
                        .build();

        Repository.registerRepositoryFactory(new S3RepositoryFactory(client));

        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls(
                                "s3://djl-ai/mlrepo/model/cv/image_classification/ai/djl/mxnet/mlp/0.0.1")
                        .optModelName("mlp")
                        .build();

        try (ZooModel<NDList, NDList> model = criteria.loadModel()) {
            Assert.assertEquals(model.getName(), "mlp");
        }
    }

    @Test
    public void testLoadArchiveFromS3()
            throws MalformedModelException, ModelNotFoundException, IOException {
        S3Client client =
                S3Client.builder()
                        .credentialsProvider(AnonymousCredentialsProvider.create())
                        .region(Region.US_EAST_1)
                        .build();

        Repository.registerRepositoryFactory(new S3RepositoryFactory(client));

        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls("s3://djl-ai/resources/test-models/mlp.tar.gz")
                        .optModelName("mlp")
                        .build();

        try (ZooModel<NDList, NDList> model = criteria.loadModel()) {
            Assert.assertEquals(model.getName(), "mlp");
        }
    }

    @Test
    public void testS3Repository() throws IOException {
        S3Client client =
                S3Client.builder()
                        .credentialsProvider(AnonymousCredentialsProvider.create())
                        .region(Region.US_EAST_1)
                        .build();

        Repository.registerRepositoryFactory(new S3RepositoryFactory(client));

        Repository repository =
                Repository.newInstance(
                        "s3",
                        "s3://djl-ai/mlrepo/model/cv/image_classification/ai/djl/mxnet/mlp/0.0.1");

        Assert.assertTrue(repository.isRemote());

        MRL mrl = repository.model(Application.UNDEFINED, "ai.djl.localmodelzoo", "mlp");
        Artifact artifact = repository.resolve(mrl, null);
        Assert.assertNotNull(artifact);

        repository = Repository.newInstance("s3", "s3://djl-ai/non-exists");
        artifact = repository.resolve(mrl, null);
        Assert.assertNull(artifact);
    }

    @Test
    public void testAccessDeny() {
        S3Client client =
                S3Client.builder()
                        .credentialsProvider(AnonymousCredentialsProvider.create())
                        .region(Region.US_EAST_1)
                        .build();

        Repository.registerRepositoryFactory(new S3RepositoryFactory(client));

        Repository repository = Repository.newInstance("s3", "s3://djl-not-exists/");

        List<MRL> list = repository.getResources();
        Assert.assertTrue(list.isEmpty());
    }
}

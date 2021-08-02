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

import ai.djl.repository.Repository;
import java.net.URI;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class S3RepositoryFactoryTest {

    @BeforeClass
    public void setUp() {
        System.setProperty("aws.region", "us-east-1");
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testUnsupportedProtocol() {
        S3RepositoryFactory factory = new S3RepositoryFactory();
        factory.newInstance("s3", URI.create("https://djl-not-exists/"));
    }

    @Test
    public void testS3RepositoryFactory() {
        Repository.newInstance("s3", "s3://djl-not-exists?artifact_id=mlp&model_name=mlp");
        Repository.newInstance("s3", "s3://djl-not-exists");
        Repository.newInstance("s3", "s3://djl-not-exists/?model_name=mlp");
    }
}

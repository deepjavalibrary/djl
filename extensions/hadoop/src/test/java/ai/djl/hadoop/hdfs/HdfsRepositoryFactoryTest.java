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
package ai.djl.hadoop.hdfs;

import ai.djl.repository.Repository;
import java.net.URI;
import org.testng.annotations.Test;

public class HdfsRepositoryFactoryTest {

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testUnsupportedProtocol() {
        HdfsRepositoryFactory factory = new HdfsRepositoryFactory();
        factory.newInstance("hdfs", URI.create("https://djl-not-exists/"));
    }

    @Test
    public void testHdfsRepositoryFactory() {
        Repository.newInstance("hdfs", "hdfs://localhost:6543?artifact_id=mlp&model_name=mlp");
        Repository.newInstance("hdfs", "hdfs://localhost:6543");
        Repository.newInstance("hdfs", "hdfs://localhost:6543/?model_name=mlp");
    }
}

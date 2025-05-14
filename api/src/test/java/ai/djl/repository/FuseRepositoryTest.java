/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.repository;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.PosixFilePermission;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.Set;

public class FuseRepositoryTest {

    @Test
    public void testGcsRepository() throws IOException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("GcsRepository is not supported on Windows");
        }

        Path gcsfuse = Paths.get("build/gcsfuse");
        Set<PosixFilePermission> permissions = PosixFilePermissions.fromString("rwxr-xr-x");
        Files.write(gcsfuse, new byte[0]);
        Files.setAttribute(gcsfuse, "posix:permissions", permissions);

        System.setProperty("GCSFUSE", "build/gcsfuse");
        System.setProperty("DJL_CACHE_DIR", "build/cache");
        Repository.registerRepositoryFactory(new RepositoryFactoryImpl.GcsRepositoryFactory());
        try {
            Repository repo = Repository.newInstance("gs", "gs://djl/resnet");
            Assert.assertEquals(repo.getResources().size(), 0);

            // test folder already exist
            Repository.newInstance("gs", "gs://djl/resnet");
        } finally {
            System.clearProperty("GCSFUSE");
            System.clearProperty("DJL_CACHE_DIR");
        }
    }

    @Test
    public void testS3Repository() throws IOException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("S3Repository is not supported on Windows");
        }

        Path gcsfuse = Paths.get("build/mount-s3");
        Set<PosixFilePermission> permissions = PosixFilePermissions.fromString("rwxr-xr-x");
        Files.write(gcsfuse, new byte[0]);
        Files.setAttribute(gcsfuse, "posix:permissions", permissions);

        System.setProperty("MOUNT_S3", "build/mount-s3");
        System.setProperty("DJL_CACHE_DIR", "build/cache");
        Repository.registerRepositoryFactory(new RepositoryFactoryImpl.S3RepositoryFactory());
        try {
            Repository repo = Repository.newInstance("s3", "s3://djl/resnet");
            Assert.assertEquals(repo.getResources().size(), 0);

            // test folder already exist
            Repository.newInstance("s3", "s3://djl/resnet");
        } finally {
            System.clearProperty("MOUNT_S3");
            System.clearProperty("DJL_CACHE_DIR");
        }
    }
}

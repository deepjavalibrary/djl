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

import ai.djl.Application;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.util.Utils;
import ai.djl.util.ZipUtils;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hdfs.MiniDFSCluster;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class HdfsRepositoryTest {

    private MiniDFSCluster miniDfs;

    @BeforeClass
    public void setup() throws IOException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("MiniDFSCluster doesn't wupport windows.");
        }

        System.setProperty("DJL_CACHE_DIR", "build/cache");
        String userHome = System.getProperty("user.home");
        System.setProperty("ENGINE_CACHE_DIR", userHome + "/.djl.ai");

        java.nio.file.Path dir = Paths.get("build/test/mlp");
        java.nio.file.Path zipFile = Paths.get("build/test/mlp.zip");
        java.nio.file.Path symbolFile = dir.resolve("mlp-symbol.json");
        java.nio.file.Path paramFile = dir.resolve("mlp-0000.param");
        Files.createDirectories(dir);
        if (Files.notExists(symbolFile)) {
            Files.createFile(symbolFile);
        }
        if (Files.notExists(paramFile)) {
            Files.createFile(paramFile);
        }
        if (Files.notExists(zipFile)) {
            ZipUtils.zip(dir, zipFile, true);
        }

        Configuration config = new Configuration();
        setFilePermission(config);
        MiniDFSCluster.Builder builder = new MiniDFSCluster.Builder(config);
        miniDfs = builder.numDataNodes(1).format(true).build();
        miniDfs.waitClusterUp();
        FileSystem fs = miniDfs.getFileSystem();
        fs.copyFromLocalFile(new Path(zipFile.toString()), new Path("/mlp.zip"));
        fs.copyFromLocalFile(new Path(symbolFile.toString()), new Path("/mlp/mlp-symbol.json"));
        fs.copyFromLocalFile(new Path(paramFile.toString()), new Path("/mlp/mlp-0000.param"));
    }

    @AfterClass
    public void tearDown() {
        miniDfs.shutdown();
        System.clearProperty("DJL_CACHE_DIR");
        System.clearProperty("ENGINE_CACHE_DIR");
    }

    @Test
    public void testZipFile() throws IOException {
        int port = miniDfs.getNameNodePort();
        Repository repo = Repository.newInstance("hdfs", "hdfs://localhost:" + port + "/mlp.zip");
        List<MRL> list = repo.getResources();
        Assert.assertFalse(list.isEmpty());

        Artifact artifact = repo.resolve(list.get(0), null);
        repo.prepare(artifact);
    }

    @Test
    public void testDir() throws IOException {
        int port = miniDfs.getNameNodePort();
        Repository repo = Repository.newInstance("hdfs", "hdfs://localhost:" + port + "/mlp");
        List<MRL> list = repo.getResources();
        Assert.assertFalse(list.isEmpty());

        Artifact artifact = repo.resolve(list.get(0), null);
        repo.prepare(artifact);

        Assert.assertTrue(repo.isRemote());
        Assert.assertEquals(repo.getName(), "hdfs");
        Assert.assertEquals(repo.getBaseUri().toString(), "hdfs://localhost:" + port);

        repo = Repository.newInstance("hdfs", "hdfs://localhost:" + port);
        list = repo.getResources();
        Assert.assertFalse(list.isEmpty());
    }

    @Test
    public void testAccessDeny() throws IOException {
        int port = miniDfs.getNameNodePort();
        Repository repo =
                Repository.newInstance("hdfs", "hdfs://localhost:" + port + "/non-exists");

        List<MRL> list = repo.getResources();
        Assert.assertTrue(list.isEmpty());

        MRL mrl = repo.model(Application.UNDEFINED, "ai.djl.localmodelzoo", "mlp");
        Artifact artifact = repo.resolve(mrl, null);
        Assert.assertNull(artifact);
    }

    private void setFilePermission(Configuration config) {
        try {
            Process process = Runtime.getRuntime().exec("/bin/sh -c umask");
            int rc = process.waitFor();
            if (rc != 0) {
                return;
            }

            try (InputStream is = process.getInputStream()) {
                String umask = Utils.toString(is).trim();
                int umaskBits = Integer.parseInt(umask, 8);
                int permBits = 0777 & ~umaskBits;
                String perms = Integer.toString(permBits, 8);
                config.set("dfs.datanode.data.dir.perm", perms);
            }
        } catch (IOException | InterruptedException ignore) {
            // ignore
        }
    }
}

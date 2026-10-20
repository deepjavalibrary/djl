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
package ai.djl.repository;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class ArtifactNameContainmentTest {

    private static final Path DIR = Paths.get("build/containment");

    @Test
    public void testNameStayingInsideIsResolved() throws IOException {
        Assert.assertEquals(
                AbstractRepository.resolveInside(DIR, "model.onnx"),
                DIR.resolve("model.onnx").toAbsolutePath().normalize());
        // A nested name is legal: the dir branch of save() exists to honour it.
        Assert.assertEquals(
                AbstractRepository.resolveInside(DIR, "sub/dir/model.onnx"),
                DIR.resolve("sub/dir/model.onnx").toAbsolutePath().normalize());
        // An empty name is deliberate in SimpleRepository and RpcRepository, and resolves to the
        // directory itself.
        Assert.assertEquals(
                AbstractRepository.resolveInside(DIR, ""), DIR.toAbsolutePath().normalize());
        // Interior traversal that does not escape is fine.
        Assert.assertEquals(
                AbstractRepository.resolveInside(DIR, "sub/../model.onnx"),
                DIR.resolve("model.onnx").toAbsolutePath().normalize());
    }

    @Test
    public void testNameEscapingIsRefused() {
        Assert.assertThrows(
                IOException.class, () -> AbstractRepository.resolveInside(DIR, "../escaped.onnx"));
        Assert.assertThrows(
                IOException.class,
                () -> AbstractRepository.resolveInside(DIR, "a/../../../etc/passwd"));
        // An absolute name ignores the directory entirely.
        Assert.assertThrows(
                IOException.class, () -> AbstractRepository.resolveInside(DIR, "/etc/passwd"));
        // A missing name is refused rather than resolving to the directory.
        Assert.assertThrows(IOException.class, () -> AbstractRepository.resolveInside(DIR, null));
    }
}

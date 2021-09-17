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
package ai.djl.benchmark;

import ai.djl.ndarray.types.DataType;
import java.net.MalformedURLException;
import java.nio.file.Paths;
import java.util.Map;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class BenchmarkTest {

    @Test
    public void testHelp() {
        String[] args = {"-h"};
        Benchmark.main(args);
    }

    @Test
    public void testArguments() throws ParseException, MalformedURLException {
        Options options = Arguments.getOptions();
        DefaultParser parser = new DefaultParser();

        String[] args = {
            "-p",
            "/opt/ml/resnet18_v1",
            "-s",
            "(1)s,(1)d,(1)u,(1)b,(1)i,(1)l,(1)B,(1)",
            "--model-options",
            "fp16,dlaCore=1",
            "--model-arguments",
            "width=28"
        };
        CommandLine cmd = parser.parse(options, args, null, false);
        Arguments arguments = new Arguments(cmd);
        String expected = Paths.get("/opt/ml/resnet18_v1").toUri().toURL().toString();
        Assert.assertEquals(arguments.getModelUrl(), expected);
        DataType[] types = arguments.getInputShapes().keyArray(new DataType[0]);
        Assert.assertEquals(types[0], DataType.FLOAT16);
        Assert.assertEquals(types[1], DataType.FLOAT64);
        Assert.assertEquals(types[2], DataType.UINT8);
        Assert.assertEquals(types[3], DataType.INT8);
        Assert.assertEquals(types[4], DataType.INT32);
        Assert.assertEquals(types[5], DataType.INT64);
        Assert.assertEquals(types[6], DataType.BOOLEAN);
        Assert.assertEquals(types[7], DataType.FLOAT32);

        Assert.assertThrows(
                IllegalArgumentException.class,
                () -> {
                    String[] arg = {"-p", "/opt/ml/resnet18_v1", "-s", "(1)S"};
                    CommandLine commandLine = parser.parse(options, arg, null, false);
                    new Arguments(commandLine);
                });

        Map<String, String> map = arguments.getModelOptions();
        Assert.assertEquals(map.get("dlaCore"), "1");
        Assert.assertTrue(map.containsKey("fp16"));

        Map<String, Object> modelArguments = arguments.getModelArguments();
        Assert.assertEquals(modelArguments.get("width"), "28");
    }

    @Test
    public void testBenchmark() {
        String[] args = {
            "-u", "djl://ai.djl.mxnet/resnet/0.0.1/resnet18_v1", "-s", "1,3,224,224", "-c", "2"
        };
        new Benchmark().runBenchmark(args);
    }

    @Test
    public void testMultithreadedBenchmark() {
        System.setProperty("collect-memory", "true");
        try {
            String[] args = {
                "-e",
                "MXNet",
                "-u",
                "djl://ai.djl.mxnet/resnet/0.0.1/resnet18_v1",
                "-s",
                "(1,3,224,224)f",
                "-d",
                "1",
                "-l",
                "1",
                "-c",
                "2",
                "-t",
                "-1",
                "-g",
                "-1"
            };
            Benchmark.main(args);
        } finally {
            System.clearProperty("collect-memory");
        }
    }
}

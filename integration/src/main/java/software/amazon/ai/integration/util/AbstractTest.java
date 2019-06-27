/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package software.amazon.ai.integration.util;

import java.lang.reflect.Method;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;
import software.amazon.ai.engine.Engine;

public class AbstractTest {

    private static final Logger logger = LogUtils.getLogger(AbstractTest.class);

    protected void test(Arguments arguments, int iteration) {
        List<Method> methods;
        String methodName = arguments.getMethodName();
        try {
            if (methodName != null) {
                methods = new ArrayList<>();
                methods.add(this.getClass().getMethod(methodName));
            } else {
                methods = new ArrayList<>(Arrays.asList(this.getClass().getMethods()));
            }
            for (Method method : methods) {
                for (int i = 0; i < iteration; i++) {
                    if (method.isAnnotationPresent(RunAsTest.class)) {
                        method.invoke(this);
                    }
                }
            }
        } catch (Exception e) {
            logger.error("{} test failed. Exception message={}", methodName, e.getCause());
        }
    }

    public void runTest(String[] args) {
        Options options = getOptions();
        try {
            DefaultParser parser = new DefaultParser();
            CommandLine cmd = parser.parse(options, args, null, false);
            Arguments arguments = parseArguments(cmd);

            Duration duration = Duration.ofMinutes(arguments.getDuration());
            int iteration = arguments.getIteration();

            logger.info("Running {}, iteration: {}", getClass().getSimpleName(), iteration);

            long init = System.nanoTime();
            String version = Engine.getInstance().getVersion();
            JnaUtils.getAllOpNames();
            long loaded = System.nanoTime();
            logger.info(
                    String.format(
                            "Load library %s in %.3f ms.", version, (loaded - init) / 1000000f));

            while (!duration.isNegative()) {
                long begin = System.currentTimeMillis();
                test(arguments, iteration);
                long delta = System.currentTimeMillis() - begin;
                duration = duration.minus(Duration.ofMillis(delta));
            }
        } catch (ParseException e) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.setLeftPadding(1);
            formatter.setWidth(120);
            formatter.printHelp(e.getMessage(), options);
        } catch (Throwable t) {
            logger.error("Unexpected error", t);
        }
    }

    protected Options getOptions() {
        return Arguments.getOptions();
    }

    protected Arguments parseArguments(CommandLine cmd) {
        return new Arguments(cmd);
    }
}

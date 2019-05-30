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
package com.amazon.ai.example.util;

import com.amazon.ai.engine.Engine;
import com.amazon.ai.image.Images;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.time.Duration;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;

public abstract class AbstractExample {

    private static final Logger logger = LogUtils.getLogger(AbstractExample.class);

    public AbstractExample() {}

    public void runExample(String[] args) {
        Options options = Arguments.getOptions();
        try {
            DefaultParser parser = new DefaultParser();
            CommandLine cmd = parser.parse(options, args, null, false);
            Arguments arguments = new Arguments(cmd);

            File modelDir = new File(arguments.getModelDir());
            String modelName = arguments.getModelName();
            String imageFile = arguments.getImageFile();
            Duration duration = Duration.ofMinutes(arguments.getDuration());
            int iteration = arguments.getIteration();

            logger.info("Running {}, iteration: {}", getClass().getSimpleName(), iteration);

            BufferedImage img = Images.loadImageFromFile(new File(imageFile));

            long init = System.nanoTime();
            String version = Engine.getInstance().getVersion();
            JnaUtils.getAllOpNames();
            long loaded = System.nanoTime();
            logger.info(
                    String.format(
                            "Load library %s in %.3f ms.", version, (loaded - init) / 1000000f));

            while (!duration.isNegative()) {
                long begin = System.currentTimeMillis();
                predict(modelDir, modelName, img, iteration);
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

    @SuppressWarnings("PMD.SystemPrintln")
    protected void printProgress(int iteration, int index, String message) {
        if (index == 0) {
            logger.info(String.format("Result: %s", message));
        } else {
            System.out.print(".");
            if (index % 80 == 0 || index == iteration - 1) {
                System.out.println();
            }
        }
    }

    public abstract void predict(
            File modelDir, String modelName, BufferedImage image, int iteration) throws IOException;
}

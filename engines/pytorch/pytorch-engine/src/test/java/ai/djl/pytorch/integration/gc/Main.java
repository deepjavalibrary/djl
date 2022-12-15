/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.pytorch.integration.gc;

import static ai.djl.pytorch.engine.PtNDManager.debugDumpFromSystemManager;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

/** Some poc testing code. */
public final class Main {

    private static final Logger logger = LoggerFactory.getLogger(Main.class);

    private Main() {}

    public static void main(String[] args)
            throws IOException, TranslateException, InterruptedException {

        try (NDManager baseManager = NDManager.newBaseManager(true); ) {
            try (NDManager subManager = baseManager.newSubManager()) {

                NDArray a = subManager.create(new float[] {1f});
                NDArray b = subManager.create(new float[] {2f});
                NDArray c = a.add(b);
                debugDumpFromSystemManager();

                logger.info("reference exists ...");
                logger.info("weakHashMap size: {}", baseManager.getProxyMaker().mapSize());
                a = null;
                b = null;
                c = null;
                logger.info("no reference exists, but likely not yet garbage collected ...");
                logger.info("weakHashMap size: {}", baseManager.getProxyMaker().mapSize());

                System.gc(); // just for testing - do not use in production
                TimeUnit.SECONDS.sleep(1);

                logger.info("no reference exists, and likely garbage collected ...");
                logger.info("weakHashMap size: {}", baseManager.getProxyMaker().mapSize());
                debugDumpFromSystemManager();
            }
            debugDumpFromSystemManager();
        }
        debugDumpFromSystemManager();
    }
}

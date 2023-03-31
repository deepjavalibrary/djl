/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.stream.Stream;

/** A utility class to detect number of nueron cores. */
public final class NeuronUtils {

    private static final Logger logger = LoggerFactory.getLogger(NeuronUtils.class);

    private NeuronUtils() {}

    /**
     * Returns whether Neuron runtime library is in the system.
     *
     * @return {@code true} if Neuron runtime library is in the system
     */
    public static boolean hasNeuron() {
        return getNeuronCores() > 0;
    }

    /**
     * Returns the number of NeuronCores available in the system.
     *
     * @return the number of NeuronCores available in the system
     */
    public static int getNeuronCores() {
        return getNeuronCores("/sys/devices/virtual/neuron_device/");
    }

    @SuppressWarnings("PMD.ForLoopCanBeForeach")
    static int getNeuronCores(String location) {
        Path path = Paths.get(location);
        if (!Files.exists(path)) {
            return 0;
        }
        int count = 0;
        try (Stream<Path> dev = Files.list(path)) {
            for (Iterator<Path> it = dev.iterator(); it.hasNext(); ) {
                Path dir = it.next();
                if (dir.toFile().getName().startsWith("neuron")) {
                    Stream<Path> cores = Files.list(dir);
                    count += Math.toIntExact(cores.filter(NeuronUtils::matches).count());
                    cores.close();
                }
            }
        } catch (IOException e) {
            logger.warn("Failed to list neuron cores", e);
        }
        return count;
    }

    private static boolean matches(Path p) {
        return p.getFileName().toString().startsWith("neuron_core");
    }
}

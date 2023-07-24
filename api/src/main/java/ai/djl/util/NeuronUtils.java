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
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
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
        List<String> nd = getNeuronDevices("/dev/");
        if (nd.isEmpty()) {
            return 0;
        }
        int cores = getNeuronCoresForDevice(nd.get(0));
        return nd.size() * cores;
    }

    /**
     * Returns a list of neuron device file path.
     *
     * @param location the neuron device path
     * @return a list of neuron device name
     */
    public static List<String> getNeuronDevices(String location) {
        Path path = Paths.get(location);
        if (!Files.exists(path)) {
            return Collections.emptyList();
        }
        try (Stream<Path> dev = Files.list(path)) {
            return dev.filter(p -> matches(p, "neuron"))
                    .map(p -> "/sys/devices/virtual/neuron_device/" + p.toFile().getName())
                    .collect(Collectors.toList());
        } catch (IOException e) {
            logger.warn("Failed to list neuron cores", e);
        }
        return Collections.emptyList();
    }

    /**
     * Returns the number of neuron cores per device.
     *
     * @param location the neuron device file path
     * @return the number of neuron cores
     */
    public static int getNeuronCoresForDevice(String location) {
        Path path = Paths.get(location);
        if (!Files.exists(path)) {
            return 0;
        }
        int count = 0;
        try (Stream<Path> dev = Files.list(path)) {
            return Math.toIntExact(dev.filter(p -> matches(p, "neuron_core")).count());
        } catch (IOException e) {
            logger.warn("Failed to list neuron cores", e);
        }
        return count;
    }

    private static boolean matches(Path p, String pattern) {
        return p.toFile().getName().startsWith(pattern);
    }
}

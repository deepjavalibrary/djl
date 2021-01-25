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
package ai.djl.serving.wlm;

import ai.djl.serving.Arguments;
import ai.djl.serving.util.ConfigManager;
import org.apache.commons.cli.CommandLine;
import org.testng.annotations.Test;

/**
 * 
 * test the Workload Manager
 * @author erik.bamberg@web.de
 *
 */
public class WorkLoadManagerTest {

    @Test
    public void testAddingJob() {
	ConfigManager.init(new Arguments(new CommandLine.Builder().build()));
	ConfigManager configManager=ConfigManager.getInstance();
	WorkLoadManager wlm=new WorkLoadManager(configManager);
	wlm.addJob("mockedModel", new Job(null,"TestModel",null));
    }
    
}

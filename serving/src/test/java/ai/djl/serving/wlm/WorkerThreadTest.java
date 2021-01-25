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

import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * testing a WorkerThread.
 * @author erik.bamberg@web.de
 *
 */
public class WorkerThreadTest {

    @Test
    public void testOfferingJobsReturnsFalseWhenCapacityExceeded() {
	ModelInfo info=new ModelInfo("TestModel", "http://testmodel/", null, 1);
	WorkerThread thread=new WorkerThread(1,null,null);
	Assert.assertTrue(thread.addJob(new Job(null,"TestModel",null)));
	Assert.assertFalse(thread.addJob(new Job(null,"TestModel",null)));
	
    }
    
    @Test
    public void testOfferingJobsReturnsFalseWhenWorkerIsShutdown() {
	ModelInfo info=new ModelInfo("TestModel", "http://testmodel/", null, 1);
	WorkerThread thread=new WorkerThread(1,null,null);
	thread.shutdown(WorkerState.WORKER_STOPPED);
	Assert.assertFalse(thread.addJob(new Job(null,"TestModel",null)));
	
    }
    
}

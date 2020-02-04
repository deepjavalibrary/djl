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
package ai.djl.basicdataset;

import ai.djl.repository.Repository;

/** An interface which contains datasets that are hosted on https://mlrepo.djl.ai/. */
public interface BasicDatasets {

    String DJL_REPO_URL = "https://mlrepo.djl.ai/";

    Repository REPOSITORY = Repository.newInstance("BasicDataset", DJL_REPO_URL);

    String GROUP_ID = "ai.djl.basicdataset";
}

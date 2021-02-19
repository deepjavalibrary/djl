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
package ai.djl.serving.http;

/**
 * constants string for http parameter names. use just this constant to keeep parameter naming
 * constant over the application.
 *
 * @author erik.bamberg@web.de
 */
public interface HttpRequestParameters {

    /** HTTP Paramater "synchronous". */
    public static final String SYNCHRONOUS_PARAMETER = "synchronous";
    /** HTTP Paramater "initial_workers". */
    public static final String INITIAL_WORKERS_PARAMETER = "initial_workers";
    /** HTTP Paramater "url". */
    public static final String URL_PARAMETER = "url";
    /** HTTP Paramater "batch_size". */
    public static final String BATCH_SIZE_PARAMETER = "batch_size";
    /** HTTP Paramater "model_name". */
    public static final String MODEL_NAME_PARAMETER = "model_name";
    /** HTTP Parameter "input_type". */
    public static final String INPUT_TYPE__PARAMETER = "input_type";
    /** HTTP Parameter "output_type". */
    public static final String OUTPUT_TYPE_PARAMETER = "output_type";
    /** HTTP Parameter "application". */
    public static final String APPLICATION_PARAMETER = "application";
    /** HTTP Parameter "group". */
    public static final String GROUP_PARAMETER = "group";
    /** HTTP Parameter "artifact". */
    public static final String ARTIFACT_PARAMETER = "artifact";
    /** HTTP Parameter "filter". */
    public static final String FILTER_PARAMETER = "filter";
    /** HTTP Paramater "max_batch_delay". */
    public static final String MAX_BATCH_DELAY_PARAMETER = "max_batch_delay";
    /** HTTP Paramater "max_idle_time". */
    public static final String MAX_IDLE_TIME__PARAMETER = "max_idle_time";
    /** HTTP Paramater "max_worker". */
    public static final String MAX_WORKER_PARAMETER = "max_worker";
    /** HTTP Paramater "min_worker". */
    public static final String MIN_WORKER_PARAMETER = "min_worker";
}

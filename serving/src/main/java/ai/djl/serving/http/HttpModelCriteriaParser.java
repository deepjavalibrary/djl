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

import ai.djl.Application;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.serving.loading.ModelCriteriaParser;
import ai.djl.serving.util.NettyUtils;
import io.netty.handler.codec.http.QueryStringDecoder;

/**
 * creates a criteria object to lookup for model using parameters from the HttpRequest.
 *
 * @author erik.bamberg@web.de
 */
public class HttpModelCriteriaParser extends ModelCriteriaParser<QueryStringDecoder> {

    /** {@inheritDoc}} */
    @Override
    protected ModelCriteriaParser<QueryStringDecoder>.Parameters parseInput(
            QueryStringDecoder decoder) {
        try {
            Parameters param = new Parameters();

            param.setInputType(
                    NettyUtils.getClassParameter(
                            decoder, HttpRequestParameters.INPUT_TYPE__PARAMETER, Input.class));
            param.setOutputType(
                    NettyUtils.getClassParameter(
                            decoder, HttpRequestParameters.OUTPUT_TYPE_PARAMETER, Output.class));
            param.setArtifactId(
                    NettyUtils.getParameter(
                            decoder, HttpRequestParameters.ARTIFACT_PARAMETER, null));
            param.setGroupId(
                    NettyUtils.getParameter(decoder, HttpRequestParameters.GROUP_PARAMETER, null));

            String applicationString =
                    NettyUtils.getParameter(
                            decoder, HttpRequestParameters.APPLICATION_PARAMETER, null);
            if (applicationString != null && !applicationString.isEmpty()) {
                param.setApplication(Application.of(applicationString));
            }

            param.setFilters(
                    NettyUtils.getMapParameter(
                            decoder, HttpRequestParameters.FILTER_PARAMETER, null));
            param.setModelUrl(
                    NettyUtils.getParameter(decoder, HttpRequestParameters.URL_PARAMETER, null));
            return param;
        } catch (ClassNotFoundException e) {
            throw new BadRequestException(
                    "input or output type. no class with this classname found", e);
        }
    }
}

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
package ai.djl.aws.sagemaker;

import ai.djl.ModelException;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;
import com.google.gson.reflect.TypeToken;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.core.exception.SdkClientException;
import software.amazon.awssdk.core.exception.SdkException;

public class SageMakerTest {

    @Test
    public void testDeployModel() throws IOException, ModelException {
        if (!Boolean.getBoolean("nightly") || !hasCredential()) {
            throw new SkipException("The test requires AWS credentials.");
        }

        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls("https://resources.djl.ai/test-models/mlp.tar.gz")
                        .build();
        try (ZooModel<NDList, NDList> model = criteria.loadModel()) {
            SageMaker sageMaker =
                    SageMaker.builder()
                            .setModel(model)
                            .optBucketName("djl-sm-test")
                            .optModelName("resnet")
                            .optContainerImage("125045733377.dkr.ecr.us-east-1.amazonaws.com/djl")
                            .optExecutionRole(
                                    "arn:aws:iam::125045733377:role/service-role/DJLSageMaker-ExecutionRole-20210213T1027050")
                            .build();

            sageMaker.deploy();

            byte[] image;
            Path imagePath = Paths.get("../../examples/src/test/resources/0.png");
            try (InputStream is = Files.newInputStream(imagePath)) {
                image = Utils.toByteArray(is);
            }
            String ret = new String(sageMaker.invoke(image), StandardCharsets.UTF_8);
            Type type = new TypeToken<List<Classifications.Classification>>() {}.getType();
            List<Classifications.Classification> list = JsonUtils.GSON.fromJson(ret, type);
            String className = list.get(0).getClassName();
            Assert.assertEquals(className, "0");

            sageMaker.deleteEndpoint();
            sageMaker.deleteEndpointConfig();
            sageMaker.deleteSageMakerModel();
        } catch (SdkException e) {
            throw new SkipException("Skip tests that requires permission.", e);
        }
    }

    private boolean hasCredential() {
        try {
            DefaultCredentialsProvider cp = DefaultCredentialsProvider.create();
            cp.resolveCredentials();
            return true;
        } catch (SdkClientException e) {
            return false;
        }
    }
}

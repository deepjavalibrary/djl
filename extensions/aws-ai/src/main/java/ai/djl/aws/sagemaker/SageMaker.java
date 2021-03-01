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

import ai.djl.Model;
import ai.djl.util.RandomUtils;
import ai.djl.util.Utils;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Date;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorOutputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.awssdk.core.SdkBytes;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.core.waiters.WaiterResponse;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.regions.providers.DefaultAwsRegionProviderChain;
import software.amazon.awssdk.services.iam.IamClient;
import software.amazon.awssdk.services.iam.model.AttachRolePolicyRequest;
import software.amazon.awssdk.services.iam.model.CreatePolicyRequest;
import software.amazon.awssdk.services.iam.model.CreatePolicyResponse;
import software.amazon.awssdk.services.iam.model.CreateRoleRequest;
import software.amazon.awssdk.services.iam.model.CreateRoleResponse;
import software.amazon.awssdk.services.iam.model.GetPolicyRequest;
import software.amazon.awssdk.services.iam.waiters.IamWaiter;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.CreateBucketRequest;
import software.amazon.awssdk.services.s3.model.HeadBucketRequest;
import software.amazon.awssdk.services.s3.model.HeadBucketResponse;
import software.amazon.awssdk.services.s3.model.ListBucketsRequest;
import software.amazon.awssdk.services.s3.model.ListBucketsResponse;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import software.amazon.awssdk.services.s3.model.S3Exception;
import software.amazon.awssdk.services.s3.waiters.S3Waiter;
import software.amazon.awssdk.services.sagemaker.SageMakerClient;
import software.amazon.awssdk.services.sagemaker.model.ContainerDefinition;
import software.amazon.awssdk.services.sagemaker.model.CreateEndpointConfigRequest;
import software.amazon.awssdk.services.sagemaker.model.CreateEndpointConfigResponse;
import software.amazon.awssdk.services.sagemaker.model.CreateEndpointRequest;
import software.amazon.awssdk.services.sagemaker.model.CreateEndpointResponse;
import software.amazon.awssdk.services.sagemaker.model.CreateModelRequest;
import software.amazon.awssdk.services.sagemaker.model.CreateModelResponse;
import software.amazon.awssdk.services.sagemaker.model.DeleteEndpointConfigRequest;
import software.amazon.awssdk.services.sagemaker.model.DeleteEndpointRequest;
import software.amazon.awssdk.services.sagemaker.model.DeleteModelRequest;
import software.amazon.awssdk.services.sagemaker.model.DescribeEndpointConfigRequest;
import software.amazon.awssdk.services.sagemaker.model.DescribeEndpointRequest;
import software.amazon.awssdk.services.sagemaker.model.DescribeModelRequest;
import software.amazon.awssdk.services.sagemaker.model.ProductionVariant;
import software.amazon.awssdk.services.sagemaker.model.SageMakerException;
import software.amazon.awssdk.services.sagemaker.waiters.SageMakerWaiter;
import software.amazon.awssdk.services.sagemakerruntime.SageMakerRuntimeClient;
import software.amazon.awssdk.services.sagemakerruntime.model.InvokeEndpointRequest;
import software.amazon.awssdk.services.sagemakerruntime.model.InvokeEndpointResponse;
import software.amazon.awssdk.services.sts.StsClient;

/** A utility class that help deploy model to SageMaker. */
public final class SageMaker {

    private static final Logger logger = LoggerFactory.getLogger(SageMaker.class);
    private static final char[] CHARS = "abcdefghijklmnopqrstuvwxyz1234567890.-".toCharArray();

    private SageMakerClient sageMaker;
    private SageMakerRuntimeClient smRuntime;
    private S3Client s3;
    private IamClient iam;
    private Region region;
    private Model model;
    private String modelName;
    private String bucketName;
    private String bucketPath;
    private String executionRole;
    private String containerImage;
    private String endpointConfigName;
    private String endpointName;
    private String instanceType;
    private int instanceCount;

    private SageMaker(Builder builder) {
        sageMaker = builder.sageMaker;
        smRuntime = builder.smRuntime;
        s3 = builder.s3;
        iam = builder.iam;
        model = builder.model;
        if (builder.modelName != null) {
            modelName = builder.modelName;
        } else {
            modelName = model.getName();
        }
        bucketName = builder.bucketName;
        bucketPath = builder.bucketPath;
        executionRole = builder.executionRole;
        containerImage = builder.containerImage;
        endpointConfigName = builder.endpointConfigName;
        endpointName = builder.endpointName;
        instanceType = builder.instanceType;
        instanceCount = builder.instanceCount;
        region = DefaultAwsRegionProviderChain.builder().build().getRegion();
    }

    /**
     * Deploys the model to Amazon SageMaker hosting service.
     *
     * @throws IOException if failed upload model to S3
     */
    public void deploy() throws IOException {
        DescribeEndpointRequest describeReq =
                DescribeEndpointRequest.builder().endpointName(endpointName).build();
        SageMakerWaiter waiter = sageMaker.waiter();
        if (doesEndpointExist()) {
            throw new IllegalStateException("Endpoint already exists: " + endpointName);
        }
        createEndpointConfig();

        logger.info("Creating endpoint {} ...", endpointName);
        CreateEndpointRequest req =
                CreateEndpointRequest.builder()
                        .endpointName(endpointName)
                        .endpointConfigName(endpointConfigName)
                        .build();
        CreateEndpointResponse resp = sageMaker.createEndpoint(req);
        String endpointArn = resp.endpointArn();
        waiter.waitUntilEndpointInService(describeReq);
        logger.info("SageMaker endpoint {} created: {}", endpointName, endpointArn);
    }

    /** Deletes the Amazon SageMaker endpoint. */
    public void deleteEndpoint() {
        logger.info("Deleting SageMaker endpoint {} ...", endpointName);
        DeleteEndpointRequest req =
                DeleteEndpointRequest.builder().endpointName(endpointName).build();
        sageMaker.deleteEndpoint(req);
        SageMakerWaiter waiter = sageMaker.waiter();
        DescribeEndpointRequest waitReq =
                DescribeEndpointRequest.builder().endpointName(endpointConfigName).build();
        waiter.waitUntilEndpointDeleted(waitReq);
        logger.info("SageMaker endpoint {} deleted.", endpointName);
    }

    /** Deletes the endpoint configuration. */
    public void deleteEndpointConfig() {
        DeleteEndpointConfigRequest req =
                DeleteEndpointConfigRequest.builder()
                        .endpointConfigName(endpointConfigName)
                        .build();
        sageMaker.deleteEndpointConfig(req);
        logger.info("SageMaker endpoint config {} deleted.", endpointConfigName);
    }

    /** Deletes the SageMaker model configuration. */
    public void deleteSageMakerModel() {
        DeleteModelRequest req = DeleteModelRequest.builder().modelName(modelName).build();
        sageMaker.deleteModel(req);
        logger.info("SageMaker model {} deleted.", modelName);
    }

    /**
     * Invokes the Amazon SageMaker endpoint.
     *
     * @param body the request payload
     * @return the inference response
     */
    public byte[] invoke(byte[] body) {
        InvokeEndpointRequest req =
                InvokeEndpointRequest.builder()
                        .endpointName(endpointName)
                        .body(SdkBytes.fromByteArray(body))
                        .build();
        InvokeEndpointResponse resp = smRuntime.invokeEndpoint(req);
        return resp.body().asByteArray();
    }

    private boolean doesEndpointExist() {
        try {
            DescribeEndpointRequest req =
                    DescribeEndpointRequest.builder().endpointName(endpointConfigName).build();
            sageMaker.describeEndpoint(req);
            return true;
        } catch (SageMakerException ignore) {
            return false;
        }
    }

    private void createEndpointConfig() throws IOException {
        if (doesEndpointConfigExist()) {
            throw new IllegalStateException(
                    "Endpoint config already exists: " + endpointConfigName);
        }
        createSageMakerModel();

        logger.info("Creating endpoint config {} ...", endpointConfigName);
        ProductionVariant variant =
                ProductionVariant.builder()
                        .variantName("AllTraffic")
                        .modelName(modelName)
                        .initialInstanceCount(instanceCount)
                        .initialVariantWeight(1.0f)
                        .instanceType(instanceType)
                        .build();
        CreateEndpointConfigRequest req =
                CreateEndpointConfigRequest.builder()
                        .endpointConfigName(endpointConfigName)
                        .productionVariants(variant)
                        .build();
        CreateEndpointConfigResponse resp = sageMaker.createEndpointConfig(req);
        String configArn = resp.endpointConfigArn();
        logger.info("SageMaker endpoint configure {} created: {}", endpointConfigName, configArn);
    }

    private boolean doesEndpointConfigExist() {
        try {
            DescribeEndpointConfigRequest req =
                    DescribeEndpointConfigRequest.builder()
                            .endpointConfigName(endpointConfigName)
                            .build();
            sageMaker.describeEndpointConfig(req);
            return true;
        } catch (SageMakerException ignore) {
            return false;
        }
    }

    private void createSageMakerModel() throws IOException {
        if (doesSageMakerModelExist()) {
            throw new IllegalStateException(
                    "SageMaker model already exists: " + endpointConfigName);
        }

        createBucket();

        Path dir = model.getModelPath();
        Path tarFile = tar(dir);
        String modelNameKey;
        if (bucketPath.isEmpty()) {
            modelNameKey = modelName + ".tar.gz";
        } else {
            modelNameKey = bucketPath + '/' + modelName + ".tar.gz";
        }
        String url = uploadModel(bucketName, modelNameKey, tarFile);
        Files.delete(tarFile);

        createRoleIfNeeded();
        getContainerImageArn();

        ContainerDefinition container =
                ContainerDefinition.builder().image(containerImage).modelDataUrl(url).build();
        CreateModelRequest req =
                CreateModelRequest.builder()
                        .modelName(modelName)
                        .primaryContainer(container)
                        .executionRoleArn(executionRole)
                        .build();
        CreateModelResponse resp = sageMaker.createModel(req);
        logger.info("SageMaker model {} created: {}", modelName, resp.modelArn());
    }

    private boolean doesSageMakerModelExist() {
        try {
            DescribeModelRequest req = DescribeModelRequest.builder().modelName(modelName).build();
            sageMaker.describeModel(req);
            return true;
        } catch (SageMakerException ignore) {
            return false;
        }
    }

    private Path tar(Path dir) throws IOException {
        Path tmp = Files.createTempFile("model", ".tar.gz");
        try (OutputStream os = Files.newOutputStream(tmp);
                BufferedOutputStream bos = new BufferedOutputStream(os);
                GzipCompressorOutputStream zos = new GzipCompressorOutputStream(bos);
                TarArchiveOutputStream tos = new TarArchiveOutputStream(zos)) {

            addToTar(dir, dir, tos);
            tos.finish();
        }
        return tmp;
    }

    private void addToTar(Path root, Path file, TarArchiveOutputStream tos) throws IOException {
        Path relative = root.relativize(file);
        String name = modelName + '/' + relative.toString();
        if (Files.isDirectory(file)) {
            File[] files = file.toFile().listFiles();
            if (files != null) {
                for (File f : files) {
                    addToTar(root, f.toPath(), tos);
                }
            }
        } else if (Files.isRegularFile(file)) {
            File f = file.toFile();
            TarArchiveEntry tarEntry = new TarArchiveEntry(f, name);
            tos.putArchiveEntry(tarEntry);
            Files.copy(file, tos);
            tos.closeArchiveEntry();
        }
    }

    private void createBucket() {
        if (doesBucketExist()) {
            logger.info("S3 bucket: {} already exists.", bucketName);
            return;
        }

        logger.info("Creating S3 bucket: {}", bucketName);
        CreateBucketRequest bucketRequest =
                CreateBucketRequest.builder().bucket(bucketName).build();

        s3.createBucket(bucketRequest);
        HeadBucketRequest bucketRequestWait =
                HeadBucketRequest.builder().bucket(bucketName).build();

        S3Waiter s3Waiter = s3.waiter();
        WaiterResponse<HeadBucketResponse> waiterResponse =
                s3Waiter.waitUntilBucketExists(bucketRequestWait);
        waiterResponse.matched().response().ifPresent(System.out::println);
    }

    private boolean doesBucketExist() {
        try {
            ListBucketsRequest request = ListBucketsRequest.builder().build();
            ListBucketsResponse response = s3.listBuckets(request);
            return response.buckets().stream().anyMatch(b -> b.name().equals(bucketName));
        } catch (S3Exception e) {
            logger.warn("Failed to check bucket existence", e);
            // in case doesn't have ListBucket permission, just assume the bucket doesn't exists
            return false;
        }
    }

    private String uploadModel(String bucketName, String key, Path path) {
        PutObjectRequest objectRequest =
                PutObjectRequest.builder().bucket(bucketName).key(key).build();

        s3.putObject(objectRequest, RequestBody.fromFile(path));
        String url = "s3://" + bucketName + '/' + key;
        logger.info("Model uploaded to: {}", url);
        return url;
    }

    private void createRoleIfNeeded() {
        if (executionRole == null) {
            SimpleDateFormat format = new SimpleDateFormat("yyyyMMdd'T'HHmmsss");
            String timestamp = format.format(new Date());
            String roleName = "DJLSageMaker-ExecutionRole-" + timestamp;
            String assumeRolePolicy = readPolicyDocument("assume_role_policy.json");
            CreateRoleRequest req =
                    CreateRoleRequest.builder()
                            .roleName(roleName)
                            .path("/service-role/")
                            .assumeRolePolicyDocument(assumeRolePolicy)
                            .description("DJL serving execution role for SageMaker.")
                            .build();
            CreateRoleResponse resp = iam.createRole(req);
            executionRole = resp.role().arn();

            String policy = readPolicyDocument("execution_policy.json");
            CreatePolicyRequest request =
                    CreatePolicyRequest.builder()
                            .policyName("DJLSageMaker-ExecutionPolicy-" + timestamp)
                            .policyDocument(policy)
                            .build();

            CreatePolicyResponse response = iam.createPolicy(request);
            String policyArn = response.policy().arn();

            GetPolicyRequest polRequest = GetPolicyRequest.builder().policyArn(policyArn).build();

            IamWaiter iamWaiter = iam.waiter();
            iamWaiter.waitUntilPolicyExists(polRequest);

            AttachRolePolicyRequest attachRequest =
                    AttachRolePolicyRequest.builder()
                            .roleName(roleName)
                            .policyArn("arn:aws:iam::aws:policy/AmazonSageMakerFullAccess")
                            .build();

            iam.attachRolePolicy(attachRequest);

            attachRequest =
                    AttachRolePolicyRequest.builder()
                            .roleName(roleName)
                            .policyArn(policyArn)
                            .build();
            iam.attachRolePolicy(attachRequest);
        }
    }

    private void getContainerImageArn() {
        if (containerImage == null) {
            String imageName = getContainerImageName();
            String accountId = StsClient.create().getCallerIdentity().account();
            String regionId = region.id();
            containerImage = accountId + ".dkr.ecr." + regionId + ".amazonaws.com/" + imageName;
        }
    }

    private String getContainerImageName() {
        String metadataFile = System.getenv("ECS_CONTAINER_METADATA_FILE");
        if (metadataFile == null) {
            throw new AssertionError("Not in a ECS container.");
        }
        Path path = Paths.get(metadataFile);
        try (Reader reader = Files.newBufferedReader(path)) {
            JsonElement json = JsonParser.parseReader(reader);
            return json.getAsJsonObject().get("ImageName").getAsString();
        } catch (IOException e) {
            throw new AssertionError("Failed to read container metadata.", e);
        }
    }

    private static String readPolicyDocument(String path) {
        try (InputStream is = SageMaker.class.getResourceAsStream(path)) {
            return Utils.toString(is);
        } catch (IOException e) {
            throw new AssertionError("Failed to read " + path, e);
        }
    }

    /**
     * Creates a builder to build a {@code SageMaker} instance.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** A Builder to construct a {@code SageMaker}. */
    public static final class Builder {

        Model model;
        String bucketName;
        String bucketPath = "";
        String executionRole;
        String containerImage;
        String endpointConfigName;
        String endpointName;
        String modelName;
        String instanceType = "ml.m4.xlarge";
        int instanceCount = 1;
        SageMakerClient sageMaker;
        SageMakerRuntimeClient smRuntime;
        S3Client s3;
        IamClient iam;

        Builder() {}

        /**
         * Sets the model to be deployed on Amazon SageMaker.
         *
         * @param model the model to be deployed on Amazon SageMaker
         * @return the builder
         */
        public Builder setModel(Model model) {
            this.model = model;
            return this;
        }

        /**
         * Sets the optional S3 bucket name to store the model.
         *
         * <p>If S3 bucket name is not provided, a random bucket with "djl-sm-" prefix will be
         * created
         *
         * @param bucketName the S3 bucket name to store the model
         * @return the builder
         */
        public Builder optBucketName(String bucketName) {
            this.bucketName = bucketName;
            return this;
        }

        /**
         * Sets the optional s3 path prefix where the model to be stored.
         *
         * @param bucketPath the s3 path prefix
         * @return the builder
         */
        public Builder optBucketPath(String bucketPath) {
            this.bucketPath = bucketPath;
            return this;
        }

        /**
         * Sets the optional role to execute the SageMaker endpoint.
         *
         * <p>If {@code executionRole} is not set, current aws role will be used. If no current role
         * found, a runtime exception will thrown.
         *
         * @param executionRole the role to execute the SageMaker endpoint
         * @return the builder
         */
        public Builder optExecutionRole(String executionRole) {
            this.executionRole = executionRole;
            return this;
        }

        /**
         * Sets the optional ECR image to deploy on SageMaker endpoint.
         *
         * <p>If {@code containerImage} is not set, current container image will be used. If the
         * code is not executed in an container, a runtime exception will thrown.
         *
         * @param containerImage ECR image to deploy on SageMaker endpoint
         * @return the builder
         */
        public Builder optContainerImage(String containerImage) {
            this.containerImage = containerImage;
            return this;
        }

        /**
         * Sets the optional endpoint configuration name to create.
         *
         * <p>If {@code endpointConfigName} is not set, model name will be used as configuration
         * name.
         *
         * @param endpointConfigName the endpoint configuration name to create
         * @return the builder
         */
        public Builder optEndpointConfigName(String endpointConfigName) {
            this.endpointConfigName = endpointConfigName;
            return this;
        }

        /**
         * Sets the optional endpoint name to create.
         *
         * <p>If {@code endpointName} is not set, model name will be used as endpoint name.
         *
         * @param endpointName the endpoint name to create
         * @return the builder
         */
        public Builder optEndpointName(String endpointName) {
            this.endpointName = endpointName;
            return this;
        }

        /**
         * Sets the optional model name to create.
         *
         * <p>If {@code modelName} is not set, model name will be used as model name.
         *
         * @param modelName the model name to create
         * @return the builder
         */
        public Builder optModelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        /**
         * Sets the optional instance type to launch the endpoint.
         *
         * <p>If {@code instanceType} is not set, "ml.m4.xlarge" will be used.
         *
         * @param instanceType the instance type to launch the endpoint
         * @return the builder
         */
        public Builder optInstanceType(String instanceType) {
            this.instanceType = instanceType;
            return this;
        }

        /**
         * Sets the number of instance to launch, default is 1.
         *
         * @param instanceCount the number of instance to launch, default is 1
         * @return the builder
         */
        public Builder optInstanceCount(int instanceCount) {
            this.instanceCount = instanceCount;
            return this;
        }

        /**
         * Sets the {@code SageMakerClient}.
         *
         * @param sageMakerClient the {@code SageMakerClient}
         * @return the builder
         */
        public Builder optSageMakerClient(SageMakerClient sageMakerClient) {
            this.sageMaker = sageMakerClient;
            return this;
        }

        /**
         * Sets the {@code SageMakerRuntimeClient}.
         *
         * @param sageMakerRuntimeClient the {@code SageMakerRuntimeClient}
         * @return the builder
         */
        public Builder optSageMakerRuntimeClient(SageMakerRuntimeClient sageMakerRuntimeClient) {
            this.smRuntime = sageMakerRuntimeClient;
            return this;
        }

        /**
         * Sets the {@code S3Client}.
         *
         * @param s3Client the {@code S3Client}
         * @return the builder
         */
        public Builder optS3Client(S3Client s3Client) {
            this.s3 = s3Client;
            return this;
        }

        /**
         * Sets the {@code IamClient}.
         *
         * @param iamClient the {@code IamClient}
         * @return the builder
         */
        public Builder optIamClient(IamClient iamClient) {
            this.iam = iamClient;
            return this;
        }

        /**
         * Builds the {@link SageMaker} with the provided data.
         *
         * @return an {@link SageMaker}
         */
        public SageMaker build() {
            if (model == null) {
                throw new IllegalArgumentException("Model is required.");
            }
            if (sageMaker == null) {
                sageMaker = SageMakerClient.create();
            }
            if (smRuntime == null) {
                smRuntime = SageMakerRuntimeClient.create();
            }
            if (s3 == null) {
                s3 = S3Client.create();
            }
            if (iam == null) {
                iam = IamClient.builder().region(Region.AWS_GLOBAL).build();
            }
            if (bucketName == null) {
                StringBuilder sb = new StringBuilder("djl-sm-");
                for (int i = 0; i < 8; ++i) {
                    sb.append(CHARS[RandomUtils.nextInt(CHARS.length)]);
                }
                bucketName = sb.toString();
            }
            if (bucketPath.endsWith("/")) {
                bucketPath = bucketPath.substring(0, bucketPath.length() - 1);
            }
            if (bucketPath.startsWith("/")) {
                bucketPath = bucketPath.substring(1);
            }
            if (endpointConfigName == null) {
                endpointConfigName = modelName == null ? model.getName() : modelName;
            }
            if (endpointName == null) {
                endpointName = modelName == null ? model.getName() : modelName;
            }

            return new SageMaker(this);
        }
    }
}

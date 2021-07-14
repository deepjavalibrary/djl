# DJL Serving Management API

DJL Serving provides a set of API allow user to manage models at runtime:

1. [Register a model](#register-a-model)
2. [Increase/decrease number of workers for specific model](#scale-workers)
3. [Describe a model's status](#describe-model)
4. [Unregister a model](#unregister-a-model)
5. [List registered models](#list-models)

Management API is listening on port 8080 and only accessible from localhost by default. To change the default setting, see [DJL Serving Configuration](configuration.md).

Similar as [Inference API](inference_api.md).

## Management APIs

### Register a model

`POST /models`
* url - Model url.
* model_name - the name of the model; this name will be used as {model_name} in other API as path.
  If this parameter is not present, modelName will be inferred by url.
* model_version - the version of the mode
* engine - the name of engine to load the model. The default is MXNet if the model doesn't define its engine.
* gpu_id - the GPU device id to load the model. The default is CPU (`-1').
* batch_size - the inference batch size. The default value is `1`.
* max_batch_delay - the maximum delay for batch aggregation. The default value is 100 milliseconds.
* max_idle_time - the maximum idle time before the worker thread is scaled down.
* min_worker - the minimum number of worker processes. The default value is `1`.
* max_worker - the maximum number of worker processes. The default is the same as the setting for `min_worker`.
* synchronous - whether or not the creation of worker is synchronous. The default value is true.

```bash
curl -X POST "http://localhost:8080/models?url=https%3A%2F%2Fresources.djl.ai%2Ftest-models%2Fmlp.tar.gz"

{
  "status": "Model \"mlp\" registered."
}
```

Download and load model may take some time, user can choose asynchronous call and check the status later.

The asynchronous call will return before trying to create workers with HTTP code 202:

```bash
curl -v -X POST "http://localhost:8080/models?url=https%3A%2F%2Fresources.djl.ai%2Ftest-models%2Fmlp.tar.gz&synchronous=false"

< HTTP/1.1 202 Accepted
< content-type: application/json
< x-request-id: bf998daa-892f-482b-a660-6d0447aa5a7a
< Pragma: no-cache
< Cache-Control: no-cache; no-store, must-revalidate, private
< Expires: Thu, 01 Jan 1970 00:00:00 UTC
< content-length: 56
< connection: keep-alive
< 
{
  "status": "Model \"mlp\" registration scheduled."
}
```

### Scale workers

`PUT /models/{model_name}`
or
`PUT /models/{model_name}/{version}`

* batch_size - the inference batch size. The default value is `1`.
* max_batch_delay - the maximum delay for batch aggregation. The default value is 100 milliseconds.
* max_idle_time - the maximum idle time before the worker thread is scaled down.
* min_worker - the minimum number of worker processes. The default value is `1`.
* max_worker - the maximum number of worker processes. The default is the same as the setting for `min_worker`.

Use the Scale Worker API to dynamically adjust the number of workers to better serve different inference request loads.

There are two different flavour of this API, synchronous vs asynchronous.

The asynchronous call will return immediately with HTTP code 202:

```bash
curl -v -X PUT "http://localhost:8080/models/mlp?min_worker=3"

< HTTP/1.1 202 Accepted
< content-type: application/json
< x-request-id: 74b65aab-dea8-470c-bb7a-5a186c7ddee6
< content-length: 33
< connection: keep-alive
< 
{
  "status": "Worker updated"
}
```

### Describe model

`GET /models/{model_name}`

Use the Describe Model API to get detail runtime status of a model:

```bash
curl http://localhost:8080/models/mlp

{
  "modelName": "mlp",
  "modelUrl": "https://resources.djl.ai/test-models/mlp.tar.gz",
  "minWorkers": 1,
  "maxWorkers": 1,
  "batchSize": 1,
  "maxBatchDelay": 100,
  "maxIdleTime": 60,
  "status": "Healthy",
  "loadedAtStartup": false,
  "workers": [
    {
      "id": 1,
      "startTime": "2021-07-14T09:01:17.199Z",
      "status": "READY",
      "gpu": false
    }
  ]
}
```

### Unregister a model

`DELETE /models/{model_name}`

Use the Unregister Model API to free up system resources:

```bash
curl -X DELETE http://localhost:8080/models/mlp

{
  "status": "Model \"mlp\" unregistered"
}
```

### List models

`GET /models`
* limit - (optional) the maximum number of items to return. It is passed as a query parameter. The default value is `100`.
* next_page_token - (optional) queries for next page. It is passed as a query parameter. This value is return by a previous API call.

Use the Models API to query current registered models:

```bash
curl "http://localhost:8080/models"
```

This API supports pagination:

```bash
curl "http://localhost:8080/models?limit=2&next_page_token=0"

{
  "models": [
    {
      "modelName": "mlp",
      "modelUrl": "https://resources.djl.ai/test-models/mlp.tar.gz"
    }
  ]
}
```

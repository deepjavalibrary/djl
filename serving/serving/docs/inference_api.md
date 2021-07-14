# DJL Serving Inference API

There are two types of APIs:

1. [Health check API](#health-check-api) - Check DJL Serving health status
2. [Predictions API](#predictions-api) - Make predictions API call to DJL Serving

## Health check API

`GET /ping`
DJL Serving supports a `ping` API that user can check health status:

```bash
curl http://localhost:8080/ping

{
  "health": "healthy!"
}
```

## Predictions API

`POST /predictions/{model_name}`

For each loaded model, user can make REST call to URI: /predictions/{model_name}/{version}

```bash
# Load PyTorch resent18 model:
curl -X POST "http://localhost:8080/models?url=https%3A%2F%2Fresources.djl.ai%2Fdemo%2Fpytorch%2Ftraced_resnet18.zip&engine=PyTorch"

# Download an image for testing
curl -O https://resources.djl.ai/images/kitten.jpg

curl -X POST http://localhost:8080/predictions/traced_resnet18 -T kitten.jpg

or:

curl -X POST http://localhost:8080/predictions/traced_resnet18 -F "data=@kitten.jpg"
```

The result was some JSON that told us our image likely held a tabby cat.

```json
[
  {
    "className": "n02123045 tabby, tabby cat",
    "probability": 0.40216901898384094
  },
  {
    "className": "n02123159 tiger cat",
    "probability": 0.29153719544410706
  },
  {
    "className": "n02124075 Egyptian cat",
    "probability": 0.27031397819519043
  },
  {
    "className": "n02123394 Persian cat",
    "probability": 0.007626921869814396
  },
  {
    "className": "n02127052 lynx, catamount",
    "probability": 0.004957360681146383
  }
]
```

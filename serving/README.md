# DJL - Model Server

## Overview

This module contains an universal model serving implementation.

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/doc/ai.djl/serving/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
# for Linux/macOS:
./gradlew javadoc

# for Windows:
..\..\gradlew javadoc
```
The javadocs output is built in the `build/doc/javadoc` folder.


## Installation
You can pull the server from the central Maven repository by including the following dependency:

```xml
<dependency>
    <groupId>ai.djl</groupId>
    <artifactId>serving</artifactId>
    <version>0.10.0</version>
    <scope>runtime</scope>
</dependency>
```

## Run model server

Use the following command to start model server locally:

```sh
cd serving

# for Linux/macOS:
./gradlew run

# for Windows:
..\..\gradlew run
```

The model server will be listening on port 8080.

You can also load a model for serving on start up:

```sh
./gradlew run --args="-m https://resources.djl.ai/test-models/mlp.tar.gz"
```

Open another terminal, and type the following command to test the inference REST API:

```sh
cd serving
curl -X POST http://127.0.0.1:8080/predictions/mlp -F "data=@../examples/src/test/resources/0.png"

{
  "classNames": [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
  ],
  "probabilities": [
    0.9999998807907104,
    2.026697559776025E-11,
    1.249230336952678E-7,
    2.777162111389231E-10,
    1.3042782132099973E-11,
    6.133447222333999E-11,
    7.507424681918451E-10,
    2.7874487162904416E-9,
    1.0341382195022675E-9,
    4.075440429573973E-9
  ]
}
```

### REST API


#### ping
url:	/ping

method: GET

example:

```sh
curl -X GET http://localhost:8080/ping
```

returns
json 

```sh
{
  "status": {Healthy|Partial Healthy|Unhealthy}
}
```




#### models - list loaded models
url:	/models

method: GET

example:

```sh
curl -X GET http://localhost:8080/models
```

returns
json 

```sh
{
  "models": [
    {
      "modelName": {modelName},
      "modelUrl": {urlWhereTheModelIsLoadedFrom}
    }
  ]
}
```



####	models - get model instance information
url:	/models/{modelName}

method: GET

example:

```sh
curl -X GET http://localhost:8080/models/mlp
```


returns
json 

```sh
{
  "modelName": {modelName},
  "modelUrl": {urlWhereTheModelIsLoadedFrom},
  "minWorkers": 8,
  "maxWorkers": 8,
  "batchSize": 1,
  "maxBatchDelay": 100,
  "status": {Healthy|Partial Healthy|Unhealthy},
  "loadedAtStartup": {true|false},
  "workers": [
    {
      "id": 1,
      "startTime": {ISO timestamp},
      "status": {READY|UNLOADING},
      "gpu": {true|false}
    },
    
	...
	
    {
      "id": {n},
      "startTime": {ISO timestamp},
      "status": {READY|UNLOADING},
      "gpu": {true|false}
    }
  ]
}
```

#### models - unregister a model
url:	/models/{modelName}

method: DELETE

example:
curl -X DELETE http://localhost:8080/models/mlp

returns
json 

```sh
{
  "status": "Model \"{modelName}\" unregistered"
}
```


#### models - scale model worker instances
url:	/models/{modelName}?{min_worker}={integer}&{max_worker}={integer}&{max_idle_time}={time in seconds}&{max_batch_delay}={time in ms}

- min_worker is optional
- max_worker is optional
- max_idle_time is optional. time is in seconds. the new set max_idle_time is only used by new created worker. Already created workers waiting for data during there idle time are not affected by a parameter change
- max_batch_delay is optional the max time in milliseconds to wait after automatically scaling up workers to offer the job before giving up.

method: PUT

example:

```sh
curl -X PUT "http://localhost:8080/models/mlp?min_worker=4&max_worker=12&max_idle_time=60&max_batch_delay=500"
```

returns
json 

```sh
{
  "status": "Model \"{modelName}\" worker scaled."
}
```

#### models - register model
url:	/models/?{modelName}=modelName&{min_worker}={integer}&{max_worker}={integer}&{max_idle_time}={int.seconds}



model_name	the name for the model
batchSize batchsize
max_batch_delay in milliseconds
min_worker is optional
max_worker is optional
max_idle_time is optional. time is in seconds
synchronous true/false

method: PUT

example:

```sh
curl -X PUT "http://localhost:8080/models?modelName=mlp?min_worker=4&max_worker=12&max_idle_time=60&max_batch_delay=100"
```

returns
json 

```sh
{
  
}
```


#### prediction - run a prediction using a loaded model
```sh
curl -X POST {host}/predictions/mlp -F "data=@../examples/src/test/resources/0.png"
```

## Logging
you can set the logging level on the command-line adding a parameter for the JVM

```sh
-Dai.djl.logging.level={FATAL|ERROR|WARN|INFO|DEBUG|TRACE}
```




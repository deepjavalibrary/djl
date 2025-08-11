# Docker Resources

DJL provides docker files that you can use to setup containers with the appropriate environment for certain platforms.

We recommend setting up a docker container with the provided Dockerfile when developing for the following
platforms and/or engines.

## Windows

You can use the [docker file](https://github.com/deepjavalibrary/djl/blob/master/docker/windows/Dockerfile) provided by us.
Please note that this docker will only work with Windows server 2019 by default. If you want it to work with other
versions of Windows, you need to pass the version as an argument as follows:

```bash
docker build --build-arg version=<YOUR_VERSION>
```

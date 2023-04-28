#!/usr/bin/env python
#
# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
import boto3
import contextlib
import hashlib
import os
import shutil
import tarfile
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen


def get_cache_dir(application, group_id, url):
    """Get the cache directory.

    :param application: The application.
    :param group_id: The group ID.
    :param url: The url of the file to store to the cache.
    """
    base_dir = os.environ.get("DJL_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".djl.ai"))
    h = hashlib.sha256(url.encode('UTF-8')).hexdigest()[:40]
    return os.path.join(base_dir, "cache/repo/model", application, group_id, h)


@contextlib.contextmanager
def tmpdir(suffix="", prefix="tmp"):
    """Create a temporary directory with a context manager. The file is deleted when the
    context exits.

    :param suffix: If suffix is not None, the directory will end with that suffix.
    :param prefix: If prefix is not None, the directory will begin with that prefix.
    """
    tmp = tempfile.mkdtemp(suffix=suffix, prefix=prefix)
    yield tmp
    shutil.rmtree(tmp)


def s3_download(url, path):
    """Download a file from S3.

    :param url: The S3 url of the file.
    :param path: The path to the file to download to.
    """
    url = urlparse(url)

    if url.scheme != "s3":
        raise ValueError("Expecting 's3' scheme, got: %s in %s" % (url.scheme, url))

    bucket, key = url.netloc, url.path.lstrip("/")
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, path)


def download_and_extract(url, path):
    """Download and extract a tar file.

    :param url: The url of the tar file.
    :param path: The path to the file to download to.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.listdir(path):
        with tmpdir() as tmp:
            tmp_file = os.path.join(tmp, "tar_file")
            if url.startswith("s3://"):
                s3_download(url, tmp_file)
                with tarfile.open(name=tmp_file, mode="r:gz") as t:
                    t.extractall(path=path)
            elif url.startswith("http://") or url.startswith("https://"):
                with urlopen(url) as response, open(tmp_file, 'wb') as f:
                    shutil.copyfileobj(response, f)
                with tarfile.open(name=tmp_file, mode="r:gz") as t:
                    t.extractall(path=path)

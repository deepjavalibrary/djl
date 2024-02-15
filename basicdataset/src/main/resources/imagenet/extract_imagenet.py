"""Prepare the ImageNet dataset"""
import os
import argparse
import hashlib
import requests
import tarfile
import pickle
import gzip
import subprocess
from tqdm import tqdm

_TRAIN_TAR = 'ILSVRC2012_img_train.tar'
_TRAIN_TAR_SHA1 = '43eda4fe35c1705d6606a6a7a633bc965d194284'
_VAL_TAR = 'ILSVRC2012_img_val.tar'
_VAL_TAR_SHA1 = '5f3f73da3395154b60528b2b2a2caf2374f5f178'

def download(url, path=None, overwrite=False, sha1_hash=None):
    """Download an given URL
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path

    if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print('Downloading %s from %s...'%(fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s"%url)
        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None: # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(r.iter_content(chunk_size=1024),
                                  total=int(total_length / 1024. + 0.5),
                                  unit='KB', unit_scale=False, dynamic_ncols=True):
                    f.write(chunk)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning('File {} is downloaded but the content hash does not match. ' \
                              'The repo may be outdated or download may be incomplete. ' \
                              'If the "repo_url" is overridden, consider switching to ' \
                              'the default repo.'.format(fname))

    return fname

def parse_args():
    parser = argparse.ArgumentParser(
        description='Setup the ImageNet dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', required=True,
                        help="The directory that contains downloaded tar files")
    parser.add_argument('--target-dir',
                        help="The directory to store extracted images")
    parser.add_argument('--checksum', action='store_true',
                        help="If check integrity before extracting.")
    parser.add_argument('--with-rec', action='store_true',
                        help="If build image record files.")
    parser.add_argument('--num-thread', type=int, default=1,
                        help="Number of threads to use when building image record file.")
    args = parser.parse_args()
    if args.target_dir is None:
        args.target_dir = args.download_dir
    return args

def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.

    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.

    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash

def check_file(filename, checksum, sha1):
    if not os.path.exists(filename):
        raise ValueError('File not found: '+filename)
    if checksum and not check_sha1(filename, sha1):
        raise ValueError('Corrupted file: '+filename)

def build_rec_process(img_dir, train=False, num_thread=1):
    rec_dir = os.path.abspath(os.path.join(img_dir, '../rec'))
    makedirs(rec_dir)
    prefix = 'train' if train else 'val'
    print('Building ImageRecord file for ' + prefix + ' ...')
    to_path = rec_dir

    # download lst file and im2rec script
    script_path = os.path.join(rec_dir, 'im2rec.py')
    script_url = 'https://raw.githubusercontent.com/apache/incubator-mxnet/master/tools/im2rec.py'
    download(script_url, script_path)

    lst_path = os.path.join(rec_dir, prefix + '.lst')
    lst_url = 'http://data.mxnet.io/models/imagenet/resnet/' + prefix + '.lst'
    download(lst_url, lst_path)

    # execution
    import sys
    cmd = [
        sys.executable,
        script_path,
        rec_dir,
        img_dir,
        '--recursive',
        '--pass-through',
        '--pack-label',
        '--num-thread',
        str(num_thread)
    ]
    subprocess.call(cmd)
    os.remove(script_path)
    os.remove(lst_path)
    print('ImageRecord file for ' + prefix + ' has been built!')

def is_within_directory(directory, target):
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    prefix = os.path.commonprefix([abs_directory, abs_target])
    return prefix == abs_directory

def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")
    tar.extractall(path, members, numeric_owner=numeric_owner)


def extract_train(tar_fname, target_dir, with_rec=False, num_thread=1):
    os.makedirs(target_dir)
    with tarfile.open(tar_fname) as tar:
        print("Extracting "+tar_fname+"...")
        # extract each class one-by-one
        pbar = tqdm(total=len(tar.getnames()))
        for class_tar in tar:
            pbar.set_description('Extract '+class_tar.name)
            class_fname = os.path.join(target_dir, class_tar.name)
            if not is_within_directory(target_dir, class_fname):
                raise Exception("Attempted Path Traversal in Tar File")

            tar.extract(class_tar, target_dir)
            class_dir = os.path.splitext(class_fname)[0]
            os.mkdir(class_dir)
            with tarfile.open(class_fname) as f:
                safe_extract(f, class_dir)

            os.remove(class_fname)
            pbar.update(1)
        pbar.close()
    if with_rec:
        build_rec_process(target_dir, True, num_thread)

def extract_val(tar_fname, target_dir, with_rec=False, num_thread=1):
    os.makedirs(target_dir)
    print('Extracting ' + tar_fname)
    with tarfile.open(tar_fname) as tar:
        safe_extract(tar, target_dir)

    # build rec file before images are moved into subfolders
    if with_rec:
        build_rec_process(target_dir, False, num_thread)
    # move images to proper subfolders
    val_maps_file = os.path.join(os.path.dirname(__file__), 'imagenet_val_maps.pklz')
    with gzip.open(val_maps_file, 'rb') as f:
        dirs, mappings = pickle.load(f)
    for d in dirs:
        os.makedirs(os.path.join(target_dir, d))
    for m in mappings:
        os.rename(os.path.join(target_dir, m[0]), os.path.join(target_dir, m[1], m[0]))

def main():
    args = parse_args()

    target_dir = os.path.expanduser(args.target_dir)
    if os.path.exists(target_dir):
        raise ValueError('Target dir ['+target_dir+'] exists. Remove it first')

    download_dir = os.path.expanduser(args.download_dir)
    train_tar_fname = os.path.join(download_dir, _TRAIN_TAR)
    check_file(train_tar_fname, args.checksum, _TRAIN_TAR_SHA1)
    val_tar_fname = os.path.join(download_dir, _VAL_TAR)
    check_file(val_tar_fname, args.checksum, _VAL_TAR_SHA1)

    build_rec = args.with_rec
    if build_rec:
        os.makedirs(os.path.join(target_dir, 'rec'))
    extract_train(train_tar_fname, os.path.join(target_dir, 'train'), build_rec, args.num_thread)
    extract_val(val_tar_fname, os.path.join(target_dir, 'val'), build_rec, args.num_thread)

if __name__ == '__main__':
    main()

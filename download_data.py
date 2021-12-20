import shutil
import os
from os.path import join
import config
import wget, hash_check
import hashlib
from _io import BufferedReader

def dummy_fetch(*args, **kwargs):
    '''
    Doesn't fetch data; just places dummy (blank) data into data/ directory so that it can pass unit tests. Actual data
    + training/testing EC2 instances will have S3 buckets mounted (i.e., not use this function).
    Parameters
    ----------
    args
    kwargs

    Returns
    -------
    None
    '''
    dest_path = 'data'
    # pathlib.Path(dest_path).mkdir(parents=True, exist_ok=True)
    self_dir = os.path.dirname(__file__)

    shutil.copytree(join(self_dir, 'tests', 'bids_sample'), join(dest_path))
    return

def data_fetch(check_hash=True):
    '''

    Parameters
    ----------
    check_hash : bool
        Whether to check the hash of the downloaded data.
    Returns
    -------
    None
    '''
    wget.download(config.data['url'])
    filename = os.path.basename(config.data['url'])

    if(check_hash):
        print('')
        print('Checking data integrity; this may take a few minutes.')
        if(check_hash_correct(filename, config.data['encrypted_hash'])):
            print('Data verified to be correct.')
        else:
            print('There is something wrong with the data. Verify that the expected files are present.')
    return

def get_sha256(filename: str,
               block_size: int = 2**16):
    '''
    Iteratively computes the sha256 hash of an open file in chunks of size block_size. Useful for large files that
    can't be held directly in memory and fed to hashlib.
    Parameters
    ----------
    filename : str
        Path of the file to evaluate.
    block_size : int
        Size of block to read from the file; units are in bits.

    Returns
    -------
    str
        Hash of the file
    '''
    sha256 = hashlib.sha256()
    f = open(filename, 'rb')
    data = f.read(block_size)
    while(len(data) > 0):
        sha256.update(data)
        data = f.read(block_size)
    f.close()
    return sha256.hexdigest()


def check_hash_correct(filename: str,
                       expected_hash: str):
    '''
    Checks whether the input file has the expected hash; returns True if it does, False otherwise.
    Parameters
    ----------
    filename : str
        Path of the file to evaluate.
    expected_hash : str
        Expected hex hash of the file.

    Returns
    -------
    bool
    '''
    return get_sha256(filename) == expected_hash


if __name__ == '__main__':
    if(config.is_quick_test):
        dummy_fetch()
        print(
            'Warning: Data is not actually being fetched. See documentation for instructions on how to get a local copy'
            ' of the data.')
    else:
        print('Fetching data; this will take a few minutes.')
        data_fetch()
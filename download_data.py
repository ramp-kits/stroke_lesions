import shutil
import os
from os.path import join
import config
import wget, hash_check
import hashlib
import osfclient
from _io import BufferedReader
import argparse
import tempfile, tarfile

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


def download_private(user: str,
                     pword: str):
    '''
    Downloads the private dataset frmo OSF.
    Parameters
    ----------
    user : str
        OSF username. User must have been added to the private OSF project.
    pword : str
        OSF password associated with the username.

    Returns
    -------
    None
    '''
    osf_conn = osfclient.OSF(username=user, password=pword)
    proj_list = [osf_conn.project(proj_id) for proj_id in config.data['private_osf_ids']]
    store_list = [proj.storage('osfstorage') for proj in proj_list]
    tmpdir = tempfile.mkdtemp()
    for s in store_list:
        for f in s.files:
            fname = f.name
            if(fname.endswith('.tar.gz')):
                f.write_to(join(tmpdir, fname))
                tarf = tarfile.open(join(tmpdir, fname))
                tarf.extractall('./')
                tarf.close()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--private', required=False, action='store_true', default=False)
    parser.add_argument('--username', required=False, type=str)
    parser.add_argument('--password', required=False, type=str)
    pargs = parser.parse_args()
    if(config.is_quick_test):
        dummy_fetch()
        print(
            'Warning: Data is not actually being fetched. See documentation for instructions on how to get a local copy'
            ' of the data.')
    elif(pargs.private):
        print('Fetching private data; this will take a few minutes.')
        download_private(user=pargs.username, pword=pargs.password)
    else:
        print('Fetching data; this will take a few minutes.')
        data_fetch()
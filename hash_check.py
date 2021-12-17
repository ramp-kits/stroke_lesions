import hashlib
from _io import BufferedReader


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

import shutil
import os
from os.path import join

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

if __name__ == '__main__':
    dummy_fetch()
    print('Warning: Data is not actually being fetched. See documentation for instructions on how to get a local copy'
          ' of the data.')

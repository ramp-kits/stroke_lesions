import os
from osfclient.api import OSF

# this script does the same as (from terminal)
# osf -p your_password -u your_username upload local_path remote_path

DATA_DIR = 'data/test/'  # local path to the data
REMOTE_PATH = 'stroke/test'  # remote path where to store the data on OSF
PROJECT_CODE = 't4uf8'  # to find your PROJECT_CODE navigate to your OSF
# project on the web. The link will be something of this type:
# https://osf.io/t4uf8/ , here t4uf8 is the PROJECT_CODE
USERNAME = ''
PASSWORD = ''  # for uploading the data you need to give the username
# and the password of one of the project owners

# if the file already exists it will overwrite it
osf = OSF(username=USERNAME, password=PASSWORD)
project = osf.project(PROJECT_CODE)

destination = 'https://osf.io/' + PROJECT_CODE + '/'
store = project.storage('osfstorage')


def upload_recursive_to_osf():
    # here we are only using recursive
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError(f"Expected source ({DATA_DIR}) to be a directory")

    _, dir_name = os.path.split(DATA_DIR)

    idx = 1
    for root, _, files in os.walk(DATA_DIR):
        # local_source = 'data/dummy.txt'
        # assert os.path.exists(local_source)
        subdir_path = os.path.relpath(root, DATA_DIR)
        for fname in files:
            local_path = os.path.join(root, fname)

            print(f'{idx} uploading: {local_path}')
            idx += 1
            with open(local_path, 'rb') as fp:
                name = os.path.join(REMOTE_PATH, dir_name, subdir_path, fname)
                store.create_file(name, fp, force=True)

    print(f'uploaded {len(files)} files to {fname}')


if __name__ == "__main__":
    upload_recursive_to_osf()

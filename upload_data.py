import os
from osfclient.api import OSF

# this script does the same as (from terminal)
# osf -r -p your_password -u your_username upload local_path remote_path

LOCAL_PATH = 'data/'  # local path to the data
REMOTE_PATH = 'stroke'  # remote path where to store the data on OSF
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
    if not os.path.isdir(REMOTE_PATH):
        raise RuntimeError(f"Expected source ({REMOTE_PATH})"
                           "to be a directory")

    _, dir_name = os.path.split(REMOTE_PATH)

    idx = 1
    for root, _, files in os.walk(REMOTE_PATH):
        subdir_path = os.path.relpath(root, REMOTE_PATH)
        for fname in files:
            local_path = os.path.join(root, fname)

            print(f'{idx} uploading: {local_path}')
            idx += 1
            with open(local_path, 'rb') as fp:
                name = os.path.join(REMOTE_PATH, dir_name, subdir_path, fname)
                store.create_file(name, fp, force=True)

    print(f'uploaded {idx-1} files to {subdir_path}')


if __name__ == "__main__":
    upload_recursive_to_osf()

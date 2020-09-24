import os
from osfclient.api import OSF

# NOTE: we are not using the fetch_from_osf from ramp_utils.datasets because
# too many files are to be loaded (hence checking the id of each of them would
# be too time consuming)

# in the command line: osf -p t4uf8 clone temp/
# however this corresponds to the whole project. we are interested only in the
# stroke data here

# this script does the same as (from terminal)
# osf upload local_path remote_path

LOCAL_PATH = 'data/'  # local path to the data
REMOTE_PATH = 'stroke/'  # remote path where to store the data on OSF
PROJECT_CODE = 't4uf8'  # to find your PROJECT_CODE navigate to your OSF
# project on the web. The link will be something of this type:
# https://osf.io/t4uf8/ , here t4uf8 is the PROJECT_CODE

# if the file already exists it will overwrite it
# osf = OSF(username=USERNAME, password=PASSWORD)
osf = OSF()
project = osf.project(PROJECT_CODE)

destination = 'https://osf.io/' + PROJECT_CODE + '/'
store = project.storage('osfstorage')


def download_from_osf():
    file_idx = 0
    for file_ in store.files:
        # get only those files which are stored in REMOTE_PATH
        pathname = file_.path
        if REMOTE_PATH not in pathname:
            # we are not interested in this file
            continue
        # otherwise we are copying it locally
        # check if the directory tree exists and add the dirs if necessary

        dirname, filename = os.path.split(pathname)

        dirs = pathname.split('/')[1:-1]  # we are only interested in dirs
        dirs[0] = LOCAL_PATH  # we are overwriting the project name with local
        # path
        # check if all the necessary dirs exist and create them otherwise
        deeper_path = dirs[0]
        for idx, check_dir in enumerate(dirs):
            if idx > 0:
                deeper_path = os.path.join(deeper_path, check_dir)
            if not os.path.exists(deeper_path):
                os.mkdir(deeper_path)
                print(f'created {deeper_path}')

        # save the file
        save_file = os.path.join(deeper_path, filename)
        with open(save_file, "wb") as f:
            file_.write_to(f)
        idx += 1
    print(f'saved {file_idx} files to {LOCAL_PATH}')


if __name__ == "__main__":
    download_from_osf()

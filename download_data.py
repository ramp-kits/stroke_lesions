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

LOCAL_PATH = 'data'  # local path to the data
REMOTE_PATH = 'stroke/'  # remote path where to store the data on OSF
PROJECT_CODE = ''  # to find your PROJECT_CODE navigate to your OSF
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

        # do not include project name
        pathname = pathname[pathname.find(REMOTE_PATH)+len(REMOTE_PATH):]
        save_file = os.path.join(LOCAL_PATH, pathname)
        pathfile, filename = os.path.split(save_file)

        if not os.path.exists(pathfile):
            os.makedirs(pathfile)

        if not os.path.exists(save_file):
            # do not save it if the file already exists
            with open(save_file, "wb") as f:
                file_.write_to(f)
            file_idx += 1
        else:
            print(f'Skipping existing file {save_file}')
    print(f'saved {file_idx} files to {LOCAL_PATH}')


if __name__ == "__main__":
    download_from_osf()

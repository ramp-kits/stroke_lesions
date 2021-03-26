import click
import numpy as np
import os
from osfclient.api import OSF
from pathlib import Path
import shutil
import tarfile

# this script does the same as (from terminal)
# osf -r -p your_password -u your_username upload local_path remote_path

REMOTE_PATH = 'stroke'  # remote path where to store the data on OSF
# if the data in this path already exists it will be overwritten
PROJECT_CODE_PUBLIC = 't4uf8'  # to find your PROJECT_CODE navigate to your OSF
# project on the web. The link will be something of this type:
# https://osf.io/t4uf8/ , here t4uf8 is the PROJECT_CODE
PROJECT_CODE_PRIVATE = 'vw8sh'


@click.command()
@click.option(
    "--username", required=True,
    help="Your username to the private repository"
)
@click.option(
    "--password", required=True,
    help="Your password to the private repository"
)
@click.option(
    "--local_path", required=True,
    help="path where you store all the data"
)
def upload_to_osf(username, password, local_path):
    # All the data in the data folder will be:
    # 1. split to public and private data directories if not done already
    # 2. zipped to tar.gz format
    # 3. uploaded to private and public osf repositiories

    local_path = Path(local_path)
    remote_path = Path(REMOTE_PATH)
    if not local_path.is_dir():
        raise RuntimeError(f"Expected source ({local_path})"
                           "to be a directory")
    osf = OSF(username=username, password=password)

    # make sure there are private and public subdirs in your data directory
    assert (local_path / 'private').is_dir()
    assert (local_path / 'public').is_dir()

    project_codes = [PROJECT_CODE_PUBLIC, PROJECT_CODE_PRIVATE]
    project_types = ['public', 'private']

    for project_code, project_type in zip(project_codes, project_types):
        used_dir = local_path / project_type
        # check if the files are already split into train and test or
        # need to be splitted
        split_train = 0.8
        shuffle = True
        if not (local_path / project_type / 'train').is_dir():
            os.mkdir((local_path / project_type / 'train'))
            os.mkdir((local_path / project_type / 'test'))
            # if 'train' subfolder does not exist we will make the split
            t1_name = '*_T1.nii.gz'
            lesion_name = '_lesion.nii.gz'

            count_t1 = len([n_file for n_file in used_dir.glob(t1_name)])
            file_indices = list(range(0, count_t1))
            if shuffle:
                np.random.shuffle(file_indices)
            n_train = int(count_t1 * split_train)
            train_indices = file_indices[:n_train]
            test_indices = file_indices[n_train:]
            for idx, next_file in enumerate(used_dir.glob(t1_name)):
                prefix = next_file.name.split('_')[0]
                lesion_file = (used_dir / (prefix + lesion_name))
                if idx in train_indices:
                    copy_to = 'train'
                    # move to the train dir
                    pass
                elif idx in test_indices:
                    # move to the test dir
                    copy_to = 'test'
                elif next_file.is_dir():
                    continue
                else:
                    raise ReferenceError
                shutil.move(lesion_file,
                            (used_dir / copy_to / (prefix + lesion_name)))
                shutil.move(next_file,
                            (used_dir / copy_to / (prefix + t1_name[1:])))

        print(f'compressing {project_type} data')

        tar_name = local_path / (project_type + '.tar.gz')

        # add files from the given dir to your archive
        with tarfile.open(tar_name, "w:gz") as tar_handle:
            for next_file in used_dir.rglob('*'):
                if not next_file.is_file():
                    continue
                print(next_file)
                remote_name = next_file.relative_to(used_dir)
                tar_handle.add(next_file, arcname=remote_name)
        print(f'uploading {project_type} data')

        # establish the connection with the correct repo on osf
        project = osf.project(project_code)
        store = project.storage('osfstorage')

        with open(tar_name, 'rb') as fp:
            fname = remote_path / (project_type + '.tar.gz')
            store.create_file(fname, fp, force=True)
        print(f'successfully uploaded {fname} to {REMOTE_PATH}')


if __name__ == "__main__":
    upload_to_osf()

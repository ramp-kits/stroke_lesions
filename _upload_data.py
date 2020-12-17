import click
from osfclient.api import OSF
from pathlib import Path
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

    # ########################################################
    # TODO: make the split to public and private data directories
    # to have a path:
    # local_path
    #       |---public
    #       |---private
    # all the data in the public directory will be added to the
    # public repo, and from private directory to the private repo
    #
    # here the split has already been done beforehand

    # make sure there are private and public subdirs in your data directory
    assert (local_path / 'private').is_dir()
    assert (local_path / 'public').is_dir()

    project_codes = [PROJECT_CODE_PUBLIC, PROJECT_CODE_PRIVATE]
    project_types = ['public', 'private']

    for project_code, project_type in zip(project_codes, project_types):

        print(f'compressing {project_type} data')
        used_dir = local_path / project_type
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
from osfclient.api import OSF

from collections import namedtuple
import hashlib
import json
import os
from urllib import request
from osfclient.cli import upload
import osfclient as osf
from osfclient.utils import split_storage


DATA_DIR = 'data/'
PROJECT_CODE = 't4uf8'  # to find your PROJECT_CODE navigate to your OSF
# project on the web. The link will be something of this type:
# https://osf.io/t4uf8/ , here t4uf8 is the PROJECT_CODE
USERNAME = ''
PASSWORD = ''  # for uploading the data you need to give the username of
# the password of one of the project owners


osf = OSF(username=USERNAME, password=PASSWORD)
project = osf.project(PROJECT_CODE)


#Storage, remote_path = split_storage(args.destination)
destination = 'https://osf.io/t4uf8/'
storage, remote_path = split_storage(destination)
store = project.storage(storage)

local_source = 'data/dummy.txt'
assert os.path.exists(local_source)

name = os.path.join(remote_path) #, 'stroke', 'test')
                                       # 'subject_1')

with open(local_source, 'rb') as fp:
    store.create_file('teset.txt', fp) #, force=args.force, update=args.update)
from distutils.core import setup

setup(
    name='ramp_stroke',
    version='0.1',
    description='RAMP module for the Stroke Segmentation Challenge',
    author='NPNL',
    py_modules=[
        'bids_loader',
        'bids_workflow',
        'config',
        'download_data',
        'indi_reformat',
        'nii_slice',
        'prediction',
        'problem',
        'problem',
        'scoring',
        'setup'],
    url='https://github.com/AlexandreHutton/stroke',
    license='gpl-3.0')

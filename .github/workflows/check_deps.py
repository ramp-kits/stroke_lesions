"""Script to check that `requirements.txt` and `environment.yml` are synced.
This script requires `pyyaml` to read `environment.yml`. It checks that all
packages listed as dependencies in it are also present in one of
`requirements.txt` or `extra_libraries.txt`.
As there might be some discrepency between package names in `pip` and `conda`,
one can add the name of the corresponding conda package as a comment on the
same line as a requirement in `requirements.txt` to ensure proper matching.
For instance, if one add the following line in `requirements.txt`
> tensorflow-gpu  # tensorflow
it will match a dependency `tensorflow` in the `environment.yml`.
"""

import yaml


def preprocess_pip_deps(lines):

    deps = []
    for dep in lines:
        dep = dep.strip()
        if len(dep) == 0 or dep.startswith('#'):
            continue

        # If there is a comment on the same line
        # use this to declare compat with conda install
        deps.append(dep.split('#')[-1].strip())
    return deps


def assert_same_deps(deps_pip, deps_conda):
    "Check the two dependencies are the same with an explicit error message."
    deps_pip = set(deps_pip)
    deps_conda = set(deps_conda) - {'pip'}
    # For requirements fetched via git, need to add parsing:
    # environment.yml uses git+https://, requirements.txt uses git+git://
    deps_pip = fix_req_set(deps_pip)
    deps_conda = fix_req_set(deps_conda)

    missing = deps_pip.symmetric_difference(deps_conda)

    assert len(missing) == 0, (
        f"Missing dependency {deps_pip.difference(deps_conda)} in `environment.yml` and "
        f"dependencies {deps_conda.difference(deps_pip)} in `requirements.txt`"
    )
    return

def fix_req_set(req_set: set):
    '''
    Parses through the input set and replaces entries starting with 'git+git://' or 'git+https://' with entries with
    those fields removed.
    Parameters
    ----------
    req_set : set
        Set containing the requirements to fix.

    Returns
    -------
    set
        Set with entries starting with 'git+git://' or 'git+https://' removed.
    '''
    returned_set = set()
    for req in req_set:
        returned_set.add(remove_git_prefix(req))
    return returned_set

def remove_git_prefix(req_name: str) -> str:
    '''
    For strings starting with either git+git:// or git+https://, remove these and return the remainder.
    Parameters
    ----------
    req_name : str
        String from which to remove the git prefix, if present.

    Returns
    -------
    str
        String without the git prefix.
    '''
    if(req_name.startswith('git+')):
        # More generally, can use req_name[req_name.index('//')+2:], but would do more than is stated in the docstring.
        if(req_name.startswith('git+git://')):
            return req_name.replace('git+git://', '')
        elif(req_name.startswith('git+https://')):
            return req_name.replace('git+https://', '')
    return req_name


if __name__ == '__main__':

    # Load deps from envrionment.yml
    with open('environment.yml') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    deps_conda = conf['dependencies']
    deps_conda = deps_conda[:-1] + deps_conda[-1]['pip']

    deps_pip = []
    for requirement_file in ['requirements.txt', 'extra_libraries.txt']:
        with open(requirement_file) as f:
            deps_pip += preprocess_pip_deps(f.readlines())

    assert_same_deps(deps_pip, deps_conda)

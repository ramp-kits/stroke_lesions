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

    missing_conda = deps_pip - deps_conda
    missing_pip = deps_conda - deps_pip
    missing = missing_pip.union(missing_conda)

    assert len(missing) == 0, (
        f"Missing dependency {missing_conda} in `environment.yml` and"
        f"dependencies {missing_pip} in `extra_libraries.tst`"
    )


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

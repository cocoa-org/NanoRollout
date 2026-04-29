# Constants - Testing Commands
TEST_ASTROPY_PYTEST = "pytest -rA -vv -o console_output_style=classic --tb=no"
TEST_DJANGO = "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1"
TEST_DJANGO_NO_PARALLEL = "./tests/runtests.py --verbosity 2"
TEST_SEABORN = "python -m pytest --no-header -rA --color=no"
TEST_SEABORN_VERBOSE = "python -m pytest -rA --tb=long --color=no"  # useless
TEST_PYTEST = "python -m pytest -rA -vv --color=no -o console_output_style=classic --tb=no"
TEST_PYTEST_VERBOSE = "pytest -rA --tb=long"  # useless
TEST_SPHINX = "tox --current-env -epy39 -v --"
TEST_SYMPY = (
    "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose"
)
TEST_SYMPY_VERBOSE = "bin/test -C --verbose"  # useless


# Constants - Installation Specifications
SPECS_SKLEARN = {
    k: {
        "python": "3.6",
        "packages": "numpy scipy cython pytest pandas matplotlib",
        "install": "python -m pip install -v --no-use-pep517 --no-build-isolation -e .",
        "pip_packages": [
            "cython",
            "numpy==1.19.2",
            "setuptools",
            "scipy==1.5.2",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["0.20", "0.21", "0.22"]
}
SPECS_SKLEARN.update(
    {
        k: {
            "python": "3.9",
            "packages": "'numpy==1.19.2' 'scipy==1.5.2' 'cython==3.0.10' pytest 'pandas<2.0.0' 'matplotlib<3.9.0' setuptools pytest joblib threadpoolctl",
            "install": "python -m pip install -v --no-use-pep517 --no-build-isolation -e .",
            "pip_packages": ["cython", "setuptools", "numpy", "scipy"],
            "test_cmd": TEST_PYTEST,
        }
        for k in ["1.3", "1.4", "1.5", "1.6"]
    }
)

SPECS_FLASK = {
    "2.0": {
        "python": "3.9",
        "packages": "requirements.txt",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "setuptools==70.0.0",
            "Werkzeug==2.3.7",
            "Jinja2==3.0.1",
            "itsdangerous==2.1.2",
            "click==8.0.1",
            "MarkupSafe==2.1.3",
        ],
        "test_cmd": TEST_PYTEST,
    },
    "2.1": {
        "python": "3.10",
        "packages": "requirements.txt",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "setuptools==70.0.0",
            "click==8.1.3",
            "itsdangerous==2.1.2",
            "Jinja2==3.1.2",
            "MarkupSafe==2.1.1",
            "Werkzeug==2.3.7",
        ],
        "test_cmd": TEST_PYTEST,
    },
}
SPECS_FLASK.update(
    {
        k: {
            "python": "3.11",
            "packages": "requirements.txt",
            "install": "python -m pip install -e .",
            "pip_packages": [
                "setuptools==70.0.0",
                "click==8.1.3",
                "itsdangerous==2.1.2",
                "Jinja2==3.1.2",
                "MarkupSafe==2.1.1",
                "Werkzeug==2.3.7",
            ],
            "test_cmd": TEST_PYTEST,
        }
        for k in ["2.2", "2.3", "3.0", "3.1"]
    }
)

SPECS_DJANGO = {
    k: {
        "python": "3.5",
        "packages": "requirements.txt",
        "pre_install": [
            "apt-get update && apt-get install -y locales",
            "echo 'en_US UTF-8' > /etc/locale.gen",
            "locale-gen en_US.UTF-8",
        ],
        "install": "python setup.py install",
        "pip_packages": ["setuptools"],
        "eval_commands": [
            "export LANG=en_US.UTF-8",
            "export LC_ALL=en_US.UTF-8",
            "export PYTHONIOENCODING=utf8",
            "export LANGUAGE=en_US:en",
        ],
        "test_cmd": TEST_DJANGO,
    }
    for k in ["1.7", "1.8", "1.9", "1.10", "1.11", "2.0", "2.1", "2.2"]
}
SPECS_DJANGO.update(
    {
        k: {
            "python": "3.5",
            "install": "python setup.py install",
            "test_cmd": TEST_DJANGO,
        }
        for k in ["1.4", "1.5", "1.6"]
    }
)
SPECS_DJANGO.update(
    {
        k: {
            "python": "3.6",
            "packages": "requirements.txt",
            "install": "python -m pip install -e .",
            "eval_commands": [
                "sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen",
                "export LANG=en_US.UTF-8",
                "export LANGUAGE=en_US:en",
                "export LC_ALL=en_US.UTF-8",
            ],
            "test_cmd": TEST_DJANGO,
        }
        for k in ["3.0", "3.1", "3.2"]
    }
)
SPECS_DJANGO.update(
    {
        k: {
            "python": "3.8",
            "packages": "requirements.txt",
            "install": "python -m pip install -e .",
            "test_cmd": TEST_DJANGO,
        }
        for k in ["4.0"]
    }
)
SPECS_DJANGO.update(
    {
        k: {
            "python": "3.9",
            "packages": "requirements.txt",
            "install": "python -m pip install -e .",
            "test_cmd": TEST_DJANGO,
        }
        for k in ["4.1", "4.2"]
    }
)
SPECS_DJANGO.update(
    {
        k: {
            "python": "3.11",
            "packages": "requirements.txt",
            "install": "python -m pip install -e .",
            "test_cmd": TEST_DJANGO,
        }
        for k in ["5.0", "5.1", "5.2"]
    }
)
SPECS_DJANGO["1.9"]["test_cmd"] = TEST_DJANGO_NO_PARALLEL

SPECS_REQUESTS = {
    k: {
        "python": "3.9",
        "packages": "pytest",
        "install": "python -m pip install .",
        "test_cmd": TEST_PYTEST,
    }
    for k in ["0.7", "0.8", "0.9", "0.11", "0.13", "0.14", "1.1", "1.2", "2.0", "2.2"]
    + ["2.3", "2.4", "2.5", "2.7", "2.8", "2.9", "2.10", "2.11", "2.12", "2.17"]
    + ["2.18", "2.19", "2.22", "2.26", "2.25", "2.27", "2.31", "3.0"]
}

SPECS_SEABORN = {
    k: {
        "python": "3.9",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "contourpy==1.1.0",
            "cycler==0.11.0",
            "fonttools==4.42.1",
            "importlib-resources==6.0.1",
            "kiwisolver==1.4.5",
            "matplotlib==3.7.2",
            "numpy==1.25.2",
            "packaging==23.1",
            "pandas==1.3.5",  # 2.0.3
            "pillow==10.0.0",
            "pyparsing==3.0.9",
            "pytest",
            "python-dateutil==2.8.2",
            "pytz==2023.3.post1",
            "scipy==1.11.2",
            "six==1.16.0",
            "tzdata==2023.1",
            "zipp==3.16.2",
        ],
        "test_cmd": TEST_SEABORN,
    }
    for k in ["0.11"]
}
SPECS_SEABORN.update(
    {
        k: {
            "python": "3.9",
            "install": "python -m pip install -e .[dev]",
            "pip_packages": [
                "contourpy==1.1.0",
                "cycler==0.11.0",
                "fonttools==4.42.1",
                "importlib-resources==6.0.1",
                "kiwisolver==1.4.5",
                "matplotlib==3.7.2",
                "numpy==1.25.2",
                "packaging==23.1",
                "pandas==2.0.0",
                "pillow==10.0.0",
                "pyparsing==3.0.9",
                "pytest",
                "python-dateutil==2.8.2",
                "pytz==2023.3.post1",
                "scipy==1.11.2",
                "six==1.16.0",
                "tzdata==2023.1",
                "zipp==3.16.2",
            ],
            "test_cmd": TEST_SEABORN,
        }
        for k in ["0.12", "0.13", "0.14"]
    }
)

SPECS_PYTEST = {
    k: {
        "python": "3.9",
        "install": "python -m pip install -e .",
        "test_cmd": TEST_PYTEST,
    }
    for k in [
        "4.4",
        "4.5",
        "4.6",
        "5.0",
        "5.1",
        "5.2",
        "5.3",
        "5.4",
        "6.0",
        "6.2",
        "6.3",
        "7.0",
        "7.1",
        "7.2",
        "7.4",
        "8.0",
        "8.1",
        "8.2",
        "8.3",
        "8.4",
    ]
}
SPECS_PYTEST["4.4"]["pip_packages"] = [
    "atomicwrites==1.4.1",
    "attrs==23.1.0",
    "more-itertools==10.1.0",
    "pluggy==0.13.1",
    "py==1.11.0",
    "setuptools==68.0.0",
    "six==1.16.0",
]
SPECS_PYTEST["4.5"]["pip_packages"] = [
    "atomicwrites==1.4.1",
    "attrs==23.1.0",
    "more-itertools==10.1.0",
    "pluggy==0.11.0",
    "py==1.11.0",
    "setuptools==68.0.0",
    "six==1.16.0",
    "wcwidth==0.2.6",
]
SPECS_PYTEST["4.6"]["pip_packages"] = [
    "atomicwrites==1.4.1",
    "attrs==23.1.0",
    "more-itertools==10.1.0",
    "packaging==23.1",
    "pluggy==0.13.1",
    "py==1.11.0",
    "six==1.16.0",
    "wcwidth==0.2.6",
]
for k in ["5.0", "5.1", "5.2"]:
    SPECS_PYTEST[k]["pip_packages"] = [
        "atomicwrites==1.4.1",
        "attrs==23.1.0",
        "more-itertools==10.1.0",
        "packaging==23.1",
        "pluggy==0.13.1",
        "py==1.11.0",
        "wcwidth==0.2.6",
    ]
SPECS_PYTEST["5.3"]["pip_packages"] = [
    "attrs==23.1.0",
    "more-itertools==10.1.0",
    "packaging==23.1",
    "pluggy==0.13.1",
    "py==1.11.0",
    "wcwidth==0.2.6",
]
SPECS_PYTEST["5.4"]["pip_packages"] = [
    "py==1.11.0",
    "packaging==23.1",
    "attrs==23.1.0",
    "more-itertools==10.1.0",
    "pluggy==0.13.1",
]
SPECS_PYTEST["6.0"]["pip_packages"] = [
    "attrs==23.1.0",
    "iniconfig==2.0.0",
    "more-itertools==10.1.0",
    "packaging==23.1",
    "pluggy==0.13.1",
    "py==1.11.0",
    "toml==0.10.2",
]
for k in ["6.2", "6.3"]:
    SPECS_PYTEST[k]["pip_packages"] = [
        "attrs==23.1.0",
        "iniconfig==2.0.0",
        "packaging==23.1",
        "pluggy==0.13.1",
        "py==1.11.0",
        "toml==0.10.2",
    ]
SPECS_PYTEST["7.0"]["pip_packages"] = [
    "attrs==23.1.0",
    "iniconfig==2.0.0",
    "packaging==23.1",
    "pluggy==0.13.1",
    "py==1.11.0",
]
for k in ["7.1", "7.2"]:
    SPECS_PYTEST[k]["pip_packages"] = [
        "attrs==23.1.0",
        "iniconfig==2.0.0",
        "packaging==23.1",
        "pluggy==0.13.1",
        "py==1.11.0",
        "tomli==2.0.1",
    ]
for k in ["7.4", "8.0", "8.1", "8.2", "8.3", "8.4"]:
    SPECS_PYTEST[k]["pip_packages"] = [
        "iniconfig==2.0.0",
        "packaging==23.1",
        "pluggy==1.3.0",
        "exceptiongroup==1.1.3",
        "tomli==2.0.1",
    ]
SPECS_PYTEST["6.3"]["pre_install"] = ["sed -i 's/>=>=/>=/' setup.cfg"]

SPECS_MATPLOTLIB = {
    k: {
        "python": "3.11",
        "packages": "environment.yml",
        "install": "python -m pip install -e .",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && DEBIAN_FRONTEND=noninteractive apt-get install -y imagemagick ffmpeg texlive texlive-latex-extra texlive-fonts-recommended texlive-xetex texlive-luatex cm-super dvipng",
            'QHULL_URL="http://www.qhull.org/download/qhull-2020-src-8.0.2.tgz"',
            'QHULL_TAR="/tmp/qhull-2020-src-8.0.2.tgz"',
            'QHULL_BUILD_DIR="/testbed/build"',
            'wget -O "$QHULL_TAR" "$QHULL_URL"',
            'mkdir -p "$QHULL_BUILD_DIR"',
            'tar -xvzf "$QHULL_TAR" -C "$QHULL_BUILD_DIR"',
        ],
        "pip_packages": [
            "contourpy==1.1.0",
            "cycler==0.11.0",
            "fonttools==4.42.1",
            "ghostscript",
            "kiwisolver==1.4.5",
            "numpy==1.25.2",
            "packaging==23.1",
            "pillow==10.0.0",
            "pikepdf",
            "pyparsing==3.0.9",
            "python-dateutil==2.8.2",
            "six==1.16.0",
            "setuptools==68.1.2",
            "setuptools-scm==7.1.0",
            "typing-extensions==4.7.1",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["3.5", "3.6", "3.7", "3.8", "3.9"]
}
SPECS_MATPLOTLIB.update(
    {
        k: {
            "python": "3.8",
            "packages": "requirements.txt",
            "install": "python -m pip install -e .",
            "pre_install": [
                "apt-get -y update && apt-get -y upgrade && DEBIAN_FRONTEND=noninteractive apt-get install -y imagemagick ffmpeg libfreetype6-dev pkg-config texlive texlive-latex-extra texlive-fonts-recommended texlive-xetex texlive-luatex cm-super",
                'QHULL_URL="http://www.qhull.org/download/qhull-2020-src-8.0.2.tgz"',
                'QHULL_TAR="/tmp/qhull-2020-src-8.0.2.tgz"',
                'QHULL_BUILD_DIR="/testbed/build"',
                'wget -O "$QHULL_TAR" "$QHULL_URL"',
                'mkdir -p "$QHULL_BUILD_DIR"',
                'tar -xvzf "$QHULL_TAR" -C "$QHULL_BUILD_DIR"',
            ],
            "pip_packages": ["pytest", "ipython"],
            "test_cmd": TEST_PYTEST,
        }
        for k in ["3.1", "3.2", "3.3", "3.4"]
    }
)
SPECS_MATPLOTLIB.update(
    {
        k: {
            "python": "3.7",
            "packages": "requirements.txt",
            "install": "python -m pip install -e .",
            "pre_install": [
                "apt-get -y update && apt-get -y upgrade && apt-get install -y imagemagick ffmpeg libfreetype6-dev pkg-config",
                'QHULL_URL="http://www.qhull.org/download/qhull-2020-src-8.0.2.tgz"',
                'QHULL_TAR="/tmp/qhull-2020-src-8.0.2.tgz"',
                'QHULL_BUILD_DIR="/testbed/build"',
                'wget -O "$QHULL_TAR" "$QHULL_URL"',
                'mkdir -p "$QHULL_BUILD_DIR"',
                'tar -xvzf "$QHULL_TAR" -C "$QHULL_BUILD_DIR"',
            ],
            "pip_packages": ["pytest"],
            "test_cmd": TEST_PYTEST,
        }
        for k in ["3.0"]
    }
)
SPECS_MATPLOTLIB.update(
    {
        k: {
            "python": "3.5",
            "install": "python setup.py build; python setup.py install",
            "pre_install": [
                "apt-get -y update && apt-get -y upgrade && && apt-get install -y imagemagick ffmpeg"
            ],
            "pip_packages": ["pytest"],
            "execute_test_as_nonroot": True,
            "test_cmd": TEST_PYTEST,
        }
        for k in ["2.0", "2.1", "2.2", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5"]
    }
)
for k in ["3.8", "3.9"]:
    SPECS_MATPLOTLIB[k]["install"] = (
        'python -m pip install --no-build-isolation -e ".[dev]"'
    )

SPECS_SPHINX = {
    k: {
        "python": "3.9",
        "pip_packages": ["tox==4.16.0", "tox-current-env==0.0.11", "Jinja2==3.0.3"],
        "install": "python -m pip install -e .[test]",
        "pre_install": ["sed -i 's/pytest/pytest -rA/' tox.ini"],
        "test_cmd": TEST_SPHINX,
    }
    for k in ["1.5", "1.6", "1.7", "1.8", "2.0", "2.1", "2.2", "2.3", "2.4", "3.0"]
    + ["3.1", "3.2", "3.3", "3.4", "3.5", "4.0", "4.1", "4.2", "4.3", "4.4"]
    + ["4.5", "5.0", "5.1", "5.2", "5.3", "6.0", "6.2", "7.0", "7.1", "7.2"]
    + ["7.3", "7.4", "8.0", "8.1"]
}
for k in ["3.0", "3.1", "3.2", "3.3", "3.4", "3.5", "4.0", "4.1", "4.2", "4.3", "4.4"]:
    SPECS_SPHINX[k]["pre_install"].extend(
        [
            "sed -i 's/Jinja2>=2.3/Jinja2<3.0/' setup.py",
            "sed -i 's/sphinxcontrib-applehelp/sphinxcontrib-applehelp<=1.0.7/' setup.py",
            "sed -i 's/sphinxcontrib-devhelp/sphinxcontrib-devhelp<=1.0.5/' setup.py",
            "sed -i 's/sphinxcontrib-qthelp/sphinxcontrib-qthelp<=1.0.6/' setup.py",
            "sed -i 's/alabaster>=0.7,<0.8/alabaster>=0.7,<0.7.12/' setup.py",
            "sed -i \"s/'packaging',/'packaging', 'markupsafe<=2.0.1',/\" setup.py",
        ]
    )
    if k in ["4.2", "4.3", "4.4"]:
        SPECS_SPHINX[k]["pre_install"].extend(
            [
                "sed -i 's/sphinxcontrib-htmlhelp>=2.0.0/sphinxcontrib-htmlhelp>=2.0.0,<=2.0.4/' setup.py",
                "sed -i 's/sphinxcontrib-serializinghtml>=1.1.5/sphinxcontrib-serializinghtml>=1.1.5,<=1.1.9/' setup.py",
            ]
        )
    elif k == "4.1":
        SPECS_SPHINX[k]["pre_install"].extend(
            [
                (
                    "grep -q 'sphinxcontrib-htmlhelp>=2.0.0' setup.py && "
                    "sed -i 's/sphinxcontrib-htmlhelp>=2.0.0/sphinxcontrib-htmlhelp>=2.0.0,<=2.0.4/' setup.py || "
                    "sed -i 's/sphinxcontrib-htmlhelp/sphinxcontrib-htmlhelp<=2.0.4/' setup.py"
                ),
                (
                    "grep -q 'sphinxcontrib-serializinghtml>=1.1.5' setup.py && "
                    "sed -i 's/sphinxcontrib-serializinghtml>=1.1.5/sphinxcontrib-serializinghtml>=1.1.5,<=1.1.9/' setup.py || "
                    "sed -i 's/sphinxcontrib-serializinghtml/sphinxcontrib-serializinghtml<=1.1.9/' setup.py"
                ),
            ]
        )
    else:
        SPECS_SPHINX[k]["pre_install"].extend(
            [
                "sed -i 's/sphinxcontrib-htmlhelp/sphinxcontrib-htmlhelp<=2.0.4/' setup.py",
                "sed -i 's/sphinxcontrib-serializinghtml/sphinxcontrib-serializinghtml<=1.1.9/' setup.py",
            ]
        )
for k in ["7.2", "7.3", "7.4", "8.0", "8.1"]:
    SPECS_SPHINX[k]["pre_install"] += ["apt-get update && apt-get install -y graphviz"]
for k in ["8.0", "8.1"]:
    SPECS_SPHINX[k]["python"] = "3.10"

SPECS_ASTROPY = {
    k: {
        "python": "3.9",
        "install": "python -m pip install -e .[test] --verbose",
        "pip_packages": [
            "attrs==23.1.0",
            "exceptiongroup==1.1.3",
            "execnet==2.0.2",
            "hypothesis==6.82.6",
            "iniconfig==2.0.0",
            "numpy==1.25.2",
            "packaging==23.1",
            "pluggy==1.3.0",
            "psutil==5.9.5",
            "pyerfa==2.0.0.3",
            "pytest-arraydiff==0.5.0",
            "pytest-astropy-header==0.2.2",
            "pytest-astropy==0.10.0",
            "pytest-cov==4.1.0",
            "pytest-doctestplus==1.0.0",
            "pytest-filter-subpackage==0.1.2",
            "pytest-mock==3.11.1",
            "pytest-openfiles==0.5.0",
            "pytest-remotedata==0.4.0",
            "pytest-xdist==3.3.1",
            "pytest==7.4.0",
            "PyYAML==6.0.1",
            "setuptools==68.0.0",
            "sortedcontainers==2.4.0",
            "tomli==2.0.1",
        ],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["3.0", "3.1", "3.2", "4.1", "4.2", "4.3", "5.0", "5.1", "5.2", "v5.3"]
}
SPECS_ASTROPY.update(
    {
        k: {
            "python": "3.6",
            "install": "python -m pip install -e .[test] --verbose",
            "packages": "setuptools==38.2.4",
            "pip_packages": [
                "attrs==17.3.0",
                "exceptiongroup==0.0.0a0",
                "execnet==1.5.0",
                "hypothesis==3.44.2",
                "cython==0.27.3",
                "jinja2==2.10",
                "MarkupSafe==1.0",
                "numpy==1.16.0",
                "packaging==16.8",
                "pluggy==0.6.0",
                "psutil==5.4.2",
                "pyerfa==1.7.0",
                "pytest-arraydiff==0.1",
                "pytest-astropy-header==0.1",
                "pytest-astropy==0.2.1",
                "pytest-cov==2.5.1",
                "pytest-doctestplus==0.1.2",
                "pytest-filter-subpackage==0.1",
                "pytest-forked==0.2",
                "pytest-mock==1.6.3",
                "pytest-openfiles==0.2.0",
                "pytest-remotedata==0.2.0",
                "pytest-xdist==1.20.1",
                "pytest==3.3.1",
                "PyYAML==3.12",
                "sortedcontainers==1.5.9",
                "tomli==0.2.0",
            ],
            "test_cmd": TEST_ASTROPY_PYTEST,
        }
        for k in ["0.1", "0.2", "0.3", "0.4", "1.1", "1.2", "1.3"]
    }
)
for k in ["4.1", "4.2", "4.3", "5.0", "5.1", "5.2", "v5.3"]:
    SPECS_ASTROPY[k]["pre_install"] = [
        'sed -i \'s/requires = \\["setuptools",/requires = \\["setuptools==68.0.0",/\' pyproject.toml'
    ]
for k in ["v5.3"]:
    SPECS_ASTROPY[k]["python"] = "3.10"

SPECS_SYMPY = {
    k: {
        "python": "3.9",
        "packages": "mpmath flake8",
        "pip_packages": ["mpmath==1.3.0", "flake8-comprehensions"],
        "install": "python -m pip install -e .",
        "test_cmd": TEST_SYMPY,
    }
    for k in ["0.7", "1.0", "1.1", "1.10", "1.11", "1.12", "1.2", "1.4", "1.5", "1.6"]
    + ["1.7", "1.8", "1.9"]
    + ["1.10", "1.11", "1.12", "1.13", "1.14"]
}
SPECS_SYMPY.update(
    {
        k: {
            "python": "3.9",
            "packages": "requirements.txt",
            "install": "python -m pip install -e .",
            "pip_packages": ["mpmath==1.3.0"],
            "test_cmd": TEST_SYMPY,
        }
        for k in ["1.13", "1.14"]
    }
)

SPECS_PYLINT = {
    k: {
        "python": "3.9",
        "packages": "requirements.txt",
        "install": "python -m pip install -e .",
        "test_cmd": TEST_PYTEST,
    }
    for k in [
        "2.10",
        "2.11",
        "2.13",
        "2.14",
        "2.15",
        "2.16",
        "2.17",
        "2.8",
        "2.9",
        "3.0",
        "3.1",
        "3.2",
        "3.3",
        "4.0",
    ]
}
SPECS_PYLINT["2.8"]["pip_packages"] = ["pyenchant==3.2"]
SPECS_PYLINT["2.8"]["pre_install"] = [
    "apt-get update && apt-get install -y libenchant-2-dev hunspell-en-us"
]
SPECS_PYLINT.update(
    {
        k: {
            **SPECS_PYLINT[k],
            "pip_packages": ["astroid==3.0.0a6", "setuptools"],
        }
        for k in ["3.0", "3.1", "3.2", "3.3", "4.0"]
    }
)
for v in ["2.14", "2.15", "2.17", "3.0", "3.1", "3.2", "3.3", "4.0"]:
    SPECS_PYLINT[v]["nano_cpus"] = int(2e9)

SPECS_XARRAY = {
    k: {
        "python": "3.10",
        "packages": "environment.yml",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "numpy==1.23.0",
            "packaging==23.1",
            "pandas==1.5.3",
            "pytest==7.4.0",
            "python-dateutil==2.8.2",
            "pytz==2023.3",
            "six==1.16.0",
            "scipy==1.11.1",
            "setuptools==68.0.0",
            "dask==2022.8.1",
        ],
        "no_use_env": True,
        "test_cmd": TEST_PYTEST,
    }
    for k in [
        "0.12",
        "0.18",
        "0.19",
        "0.20",
        "2022.03",
        "2022.06",
        "2022.09",
        "2023.07",
        "2024.05",
    ]
}

SPECS_SQLFLUFF = {
    k: {
        "python": "3.9",
        "packages": "requirements.txt",
        "install": "python -m pip install -e .",
        "test_cmd": TEST_PYTEST,
    }
    for k in [
        "0.10",
        "0.11",
        "0.12",
        "0.13",
        "0.4",
        "0.5",
        "0.6",
        "0.8",
        "0.9",
        "1.0",
        "1.1",
        "1.2",
        "1.3",
        "1.4",
        "2.0",
        "2.1",
        "2.2",
    ]
}

SPECS_DBT_CORE = {
    k: {
        "python": "3.9",
        "packages": "requirements.txt",
        "install": "python -m pip install -e .",
    }
    for k in [
        "0.13",
        "0.14",
        "0.15",
        "0.16",
        "0.17",
        "0.18",
        "0.19",
        "0.20",
        "0.21",
        "1.0",
        "1.1",
        "1.2",
        "1.3",
        "1.4",
        "1.5",
        "1.6",
        "1.7",
    ]
}

SPECS_PYVISTA = {
    k: {
        "python": "3.9",
        "install": "python -m pip install -e .",
        "pip_packages": ["pytest"],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["0.20", "0.21", "0.22", "0.23"]
}
SPECS_PYVISTA.update(
    {
        k: {
            "python": "3.9",
            "packages": "requirements.txt",
            "install": "python -m pip install -e .",
            "pip_packages": ["pytest"],
            "test_cmd": TEST_PYTEST,
            "pre_install": [
                "apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libxrender1"
            ],
        }
        for k in [
            "0.24",
            "0.25",
            "0.26",
            "0.27",
            "0.28",
            "0.29",
            "0.30",
            "0.31",
            "0.32",
            "0.33",
            "0.34",
            "0.35",
            "0.36",
            "0.37",
            "0.38",
            "0.39",
            "0.40",
            "0.41",
            "0.42",
            "0.43",
        ]
    }
)

SPECS_ASTROID = {
    k: {
        "python": "3.9",
        "install": "python -m pip install -e .",
        "pip_packages": ["pytest"],
        "test_cmd": TEST_PYTEST,
    }
    for k in [
        "2.10",
        "2.12",
        "2.13",
        "2.14",
        "2.15",
        "2.16",
        "2.5",
        "2.6",
        "2.7",
        "2.8",
        "2.9",
        "3.0",
    ]
}

SPECS_MARSHMALLOW = {
    k: {
        "python": "3.9",
        "install": "python -m pip install -e '.[dev]'",
        "test_cmd": TEST_PYTEST,
    }
    for k in [
        "2.18",
        "2.19",
        "2.20",
        "3.0",
        "3.1",
        "3.10",
        "3.11",
        "3.12",
        "3.13",
        "3.15",
        "3.16",
        "3.19",
        "3.2",
        "3.4",
        "3.8",
        "3.9",
    ]
}

SPECS_PVLIB = {
    k: {
        "python": "3.9",
        "install": "python -m pip install -e .[all]",
        "packages": "pandas scipy",
        "pip_packages": ["jupyter", "ipython", "matplotlib", "pytest", "flake8"],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
}

SPECS_PYDICOM = {
    k: {
        "python": "3.6",
        "install": "python -m pip install -e .",
        "packages": "numpy",
        "pip_packages": ["pytest"],
        "test_cmd": TEST_PYTEST,
    }
    for k in [
        "1.0",
        "1.1",
        "1.2",
        "1.3",
        "1.4",
        "2.0",
        "2.1",
        "2.2",
        "2.3",
        "2.4",
        "3.0",
    ]
}
SPECS_PYDICOM.update({k: {**SPECS_PYDICOM[k], "python": "3.8"} for k in ["1.4", "2.0"]})
SPECS_PYDICOM.update({k: {**SPECS_PYDICOM[k], "python": "3.9"} for k in ["2.1", "2.2"]})
SPECS_PYDICOM.update({k: {**SPECS_PYDICOM[k], "python": "3.10"} for k in ["2.3"]})
SPECS_PYDICOM.update(
    {k: {**SPECS_PYDICOM[k], "python": "3.11"} for k in ["2.4", "3.0"]}
)

SPECS_HUMANEVAL = {k: {"python": "3.9", "test_cmd": "python"} for k in ["1.0"]}


# --------------------------------------------------------------------------- #
# SWE-Gym
# --------------------------------------------------------------------------- #

# mypy and python versoin are tightly coupled
SPECS_MYPY = {
    k: {
        "pre_install": [
            "git submodule update --init mypy/typeshed || true",
        ],
        "python": "3.12", 
        # see https://github.com/python/mypy/mypy/test/testcheck.py#L39
        "install": "python -m pip install -r test-requirements.txt; python -m pip install -e .; hash -r",
        "test_cmd": "pytest -rA -k"
    }
    for k in ["1.7","1.8","1.9", "1.10", "1.11"]
}

SPECS_MYPY.update(
    # Working
    {
        k: {
            "pre_install": [
                "git submodule update --init mypy/typeshed || true",
            ],
            "python": "3.11", 
            "install": "python -m pip install -r test-requirements.txt; python -m pip install -e .; hash -r",
            "test_cmd": "pytest -n0 -rA -k"
        }
        for k in ["1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6"]
    }
)

SPECS_MYPY.update(
    # Working
    {
        k: {
            "pre_install": [
                "git submodule update --init mypy/typeshed || true",
            ],
            "python": "3.10", 
            "install": "python -m pip install -r test-requirements.txt; python -m pip install -e .; pip install pytest pytest-xdist; hash -r",
            "test_cmd": "pytest -n0 -rA -k"
        }
        for k in ["0.990", "0.980", "0.970", "0.960","0.950", "0.940"]
    }
)
SPECS_MYPY.update(
    # Working
    {
        k: {
            "pre_install": [
                "git submodule update --init mypy/typeshed || true",
                "sed -i '1i types-typing-extensions==3.7.3' test-requirements.txt"
            ],
            "python": "3.9", 
            # types-typing-extensions is yanked, we need to set a specific version manually
            "install": "python -m pip install -r test-requirements.txt; python -m pip install -e .; pip install pytest pytest-xdist; hash -r;",
            "test_cmd": "pytest -n0 -rA -k"
        }
        for k in ["0.920", "0.910", "0.820", "0.810", "0.800"]
    }
)

# python/mypy versions prior to 0.800 are hard to install, skipping for now
# SPECS_MYPY.update(
#     {
#         k: {
#             "pre_install": [
#                 "apt-get -y update && apt-get -y upgrade && apt-get install -y gcc",
#                 "apt-get install libxml2-dev libxslt1-dev"
#             ],
#             "python": "3.8", 
#                 "apt-get update && apt-get install -y libenchant-2-dev hunspell-en-us"
#             "install": "python -m pip install -r test-requirements.txt; python -m pip install -e .; pip install pytest; hash -r;",
#             "test_cmd": "pytest -rA -k"
#         }
#         for k in []
#     }
# )
# mypy 0.2, with 14 instances, is too old and requires deprecated python 3.4. 
# not worth it for now

TEST_MOTO = "pytest -n0 -rA"
SPECS_MOTO = {
    k: {
        "python": "3.12", 
        # see https://github.com/getmoto/moto/blob/master/CONTRIBUTING.md
        "install": "make init",
        "test_cmd": TEST_MOTO,
    }
    for k in [
        '0.4', '1.0', '1.2', '1.3',
        '2.0', '2.1', '2.2', '2.3',
        '3.0', '3.1',
        '4.0', '4.1', '4.2', '5.0',
    ]
}

TEST_CONAN = "pytest -n0 -rA"


# extra args before cython3.0 https://github.com/conan-io/conan/issues/14319
SPECS_CONAN = {
    k: {
        "python": "3.10", 
            "pre_install": [
                "apt-get -y update && apt-get -y upgrade && apt-get install -y build-essential cmake",
            ],

        "install": "echo 'cython<3' > /tmp/constraint.txt; export PIP_CONSTRAINT=/tmp/constraint.txt; python -m pip install -r conans/requirements.txt; python -m pip install -r conans/requirements_server.txt; python -m pip install -r conans/requirements_dev.txt ",
        "eval_commands": [
            "export PYTHONPATH=${PYTHONPATH:-}:$(pwd)",
        ],
        "test_cmd": TEST_CONAN,
    }
    for k in ['1.33', '1.34', '1.36', '2.0', '1.35', '1.37', '1.46', '1.38', '1.39', '1.40', '1.41', '1.42', '1.45', '1.43', '1.44', '1.47', '1.48', '1.49', '1.50', '1.51', '1.52', '1.53', '1.55', '1.54', '1.57', '1.58', '1.59']
}
 
SPECS_CONAN.update({
    k: {
        "python": "3.10", 
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y build-essential cmake",
        ],
        "install": "python -m pip install -r conans/requirements.txt; python -m pip install -r conans/requirements_server.txt; python -m pip install -r conans/requirements_dev.txt ",
        "eval_commands": [
            "export PYTHONPATH=${PYTHONPATH:-}:$(pwd)",
        ],
        "test_cmd": TEST_CONAN,
    }
    for k in ['2.1', '1.60', '1.61', '1.62', '2.2', '2.3', '2.4']
})

TEST_DASK = "pytest -n0 -rA  --color=no"
# pandas 2.0 is a breaking change, need to separate from there
SPECS_DASK = {
    k: {
        # "python": "3.10", 
        "env_patches": [
            # dask installs latest dask from github in environment.yml
            # remove these lines and delay dask installation later
            "sed -i '/- pip:/,/^ *-/d' environment.yml"
        ],
        "packages": "environment.yml",
        "install": 'python -m pip install --no-deps -e .',
        "test_cmd": TEST_DASK,
    }
    for k in ['2.11', '2.12', '2.13', '2.14', '2.15', '2.16', '2.17', '2.18', '2.19', '2.21', '2.22', '2.23', '2.25', '2.26', '2.27', '2.28', '2.29', '2.30', '2020.12', '2021.01', '2021.02', '2021.03', '2021.04', '2021.05', '2021.06', '2021.07', '2021.08', '2021.09', '2021.10', '2021.11', '2021.12', '2022.01', '2022.02', '2022.03', '2022.04', '2022.05', '2022.6', '2022.7', '2022.8', '2022.9', '2022.10', '2022.11', '2022.12', '2023.1', '2023.2', '2023.3', '2023.4', '2023.5', '2023.6', '2023.7', '2023.8', '2023.9', '2023.10', '2023.11', '2023.12', '2024.1', '2024.2', '2024.3', '2024.4', '2024.5']
}

TEST_MONAI = "pytest -rA "
SPECS_MONAI = {
    k: {
        "python": "3.8", 
        # monai's requirements.txt calls each other, hard to standardize in swebench constant format
        # "packages": "requirements.txt",
        # "install": "python -m pip install -U pip; python -m pip install scikit-build; python -m pip install types-pkg-resources==0.1.3 pytest; python -m pip install -U -r requirements-dev.txt; python setup.py develop;",
        # "env_patches": [
        #     # monai installs itself from git 
        #     # remove these lines and delay dask installation later
        #     "sed -i '/^git+https:\/\/github.com\/Project-MONAI\//d' ~/requirements.txt"
        # ],
        "install": "sed -i '/^git+https:\/\/github.com\/Project-MONAI\//d' requirements-dev.txt; python -m pip install types-pkg-resources==0.1.3 pytest; pip install -r requirements-dev.txt;python setup.py develop;",
        "test_cmd": TEST_MONAI,
    }
    for k in ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.11', '0.105', '1.0', '1.1', '1.2', '1.3']
}

# dvc
TEST_DVC = "pytest -rA"
SPECS_DVC = {
    k: {
        "python": "3.10", 
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y cmake",
            # fix moto dev version missing issue
            "[ -f setup.py ] && sed -E -i 's/moto==([0-9]+\.[0-9]+\.[0-9]+)\.dev[0-9]+/moto==\\1/' setup.py",
            # fix pyarrow version issue
            "[ -f setup.py ] && sed -i 's/pyarrow==0.15.1/pyarrow==0.16/' setup.py"
            # fix boto version conflict
            "[ -f setup.py ] && sed -i 's/boto3==1.9.115/boto3==1.9.201/' setup.py"
        ],
        "install": 'python -m pip install --upgrade pip wheel GitPython; python -m pip install "cython<3.0.0" && python -m pip install --no-build-isolation pyyaml==5.4.1; python -m pip install git+https://github.com/iterative/mock-ssh-server.git || true; python -m pip install -r tests/requirements.txt || true; python -m pip install -r test-requirements.txt || true; python -m pip install -e ".[tests,dev,all_remotes,all,testing]";',
        "test_cmd": TEST_DVC,
    }
    for k in ['0.1', '0.8', '0.9', '0.12', '0.13', '0.14', '0.15', '0.16', '0.17', '0.18', '0.19', '0.20', '0.21', '0.22', '0.23', '0.24', '0.27', '0.28', '0.29', '0.30', '0.31', '0.32', '0.33', '0.34', '0.35', '0.40', '0.41', '0.50', '0.51', '0.52', '0.53', '0.54', '0.55', '0.56', '0.57', '0.58', '0.59', '0.60', '0.61', '0.62', '0.63', '0.65', '0.66', '0.68', '0.69', '0.70', '0.71', '0.74', '0.75', '0.76', '0.77', '0.78', '0.80', '0.81', '0.82', '0.83', '0.84', '0.85', '0.86', '0.87', '0.88', '0.89', '0.90', '0.91', '0.92', '0.93', '1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '1.10', '1.11', '2.0', '2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7', '2.8', '2.9', '2.10', '2.11', '2.12', '2.13', '2.15', '2.17', '2.19', '2.20', '2.21', '2.22', '2.23', '2.24', '2.27', '2.28', '2.30', '2.33', '2.34', '2.35', '2.38', '2.41', '2.43', '2.44', '2.45', '2.46', '2.48', '2.50', '2.51', '2.52', '2.54', '2.55', '2.56', '2.57', '2.58', '3.0', '3.1', '3.2', '3.3', '3.4', '3.5', '3.6', '3.10', '3.11', '3.12', '3.13', '3.14', '3.15', '3.17', '3.19', '3.23', '3.24', '3.28', '3.29', '3.36', '3.37', '3.38', '3.43', '3.47', '3.48', '3.49']
}
for k in [
    '0.1', '0.8', '0.9', '0.12', '0.13', '0.14', '0.15', '0.16', '0.17', '0.18', '0.19', '0.20', '0.21', '0.22', '0.23', '0.24', '0.27', '0.28', '0.29', '0.30', '0.31', '0.32', '0.33', '0.34', '0.35', '0.40', '0.41', '0.50', '0.51', '0.52', '0.53', '0.54', '0.55', '0.56', '0.57', '0.58', '0.59', '0.60', '0.61', '0.62', '0.63', '0.65', '0.66', '0.68', '0.69', '0.70', '0.71', '0.74', '0.75', '0.76', '0.77', '0.78', '0.80', '0.81', '0.82', '0.83', '0.84', '0.85', '0.86', '0.87', '0.88', '0.89', '0.90', '0.91', '0.92', '0.93', ]:
    SPECS_DVC[k]['python'] = '3.8'
    SPECS_DVC[k]['install'] += ' python -m pip install "numpy<=1.20";'
    # pytest 8 breaks pytest-lazy-fixture
    SPECS_DVC[k]['install'] += ' python -m pip install "pytest<8";'

for k in [
    '1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '1.10', '1.11', '2.0', '2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7', '2.8', '2.9', '2.10', '2.11', '2.12', '2.13', '2.15', '2.17', '2.19', '2.20', '2.21', '2.22', '2.23', '2.24', '2.27', '2.28', '2.30', '2.33', '2.34', '2.35', '2.38', '2.41', '2.43', '2.44', '2.45', '2.46', '2.48', '2.50', '2.51', '2.52', '2.54', '2.55', '2.56', '2.57', '2.58', '3.0', '3.1', '3.2', '3.3', 
]:
    SPECS_DVC[k]['python'] = '3.9'
    SPECS_DVC[k]['install'] += ' python -m pip install "numpy<=1.20";'
    # pytest 8 breaks pytest-lazy-fixture
    SPECS_DVC[k]['install'] += ' python -m pip install "pytest<8";'


# bokeh
# https://docs.bokeh.org/en/latest/docs/dev_guide/setup.html
TEST_BOKEH = "pytest -rA -n0"
    # for k in ['0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.10', '0.11', '0.12', '0.13', '0.1181316818', '1.0', '1.1', '1.2', '1.3', '1.4', '2.0', '2.1', '2.3', '2.4', '3.0', '3.3', '3.4', '3.5']
SPECS_BOKEH = {
    k: {
        "python": "3.10", 
        "packages": "environment.yml",
        "pre_install": [
            "cd bokehjs && npm install --location=global npm && npm ci && cd ../"
        ],
        "install": "python -m pip install -e .; python -m pip install bokeh_sampledata;",
        "test_cmd": TEST_BOKEH,
    }
    for k in ['3.0', '3.3', '3.4', '3.5']
}

SPECS_BOKEH.update({
    k: {
        "python": "3.8", 
        "packages": "environment.yml",
        "env_patches": [
            ": \"${CONDA_MKL_INTERFACE_LAYER_BACKUP:=''}\"",
            # "sed -i 's/  - setuptools/  - setuptools<66/' environment.yml"
        ],
        "pre_install": [
            "cd bokehjs && npm install --location=global npm && npm ci && cd ../",
        ],
        "install": 'pip install "setuptools<66" "jinja2<3.1"; printf "1\n" | python setup.py develop; bokeh sampledata;',
        "test_cmd": TEST_BOKEH,
    }
    for k in ['2.0', '2.1', '2.3', '2.4']
})

SPECS_BOKEH.update({
    k: {
        "python": "3.8", 
        "packages": "environment.yml",
        "env_patches": [
            ": \"${CONDA_MKL_INTERFACE_LAYER_BACKUP:=''}\"",
            # "sed -i 's/  - setuptools/  - setuptools<66/' environment.yml"
        ],
        "pre_install": [
            "cd bokehjs && npm install --location=global npm && npm ci && cd ../",
        ],
        "install": 'pip install "setuptools<66" "jinja2<3.1"; printf "1\n" | python setup.py develop; bokeh sampledata;',
        "test_cmd": TEST_BOKEH,
    }
    for k in ['0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.10', '0.11', '0.12', '0.13', '0.1181316818', '1.0', '1.1', '1.2', '1.3', '1.4']
})

# modin
# https://github.com/modin-project/modin/pull/7312
# numpy2.0 is supported in June 2024, we will need to restrict numpy version to be before 2.0
TEST_MODIN = "pytest -n0 -rA"
SPECS_MODIN = {
    k: {
        "python": "3.9", 
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y libpq-dev",
        ],
        "packages": "environment.yml",
        "install": "python -m pip install -e .;",
        # "install": "python -m pip install 'numpy<2.0'; python -m pip install --upgrade Cython; python -m pip install -r requirements-dev.txt; python -m pip install -e .",
        "test_cmd": TEST_MODIN,
    }
    for k in ['0.1', '0.2', '0.3', '0.4', '0.6', '0.8', '0.9', '0.10', '0.11', '0.12', '0.13', '0.14', '0.15', '0.16', '0.17', '0.18', '0.19', '0.20', '0.21', '0.22', '0.23', '0.24', '0.25', '0.26', '0.27', '0.28', '0.29', '0.30']
}
for k in  ['0.1', '0.2', '0.3', '0.4', '0.6', '0.8', '0.9', '0.10', '0.11', '0.12', '0.13', '0.14', '0.15', '0.16', '0.17', '0.18', '0.19']:
    SPECS_MODIN[k]['python'] = '3.8'
    SPECS_MODIN[k]['install'] += ' python -m pip install numpy==1.23.1 protobuf==3.20.1;'

# spyder
# https://github.com/spyder-ide/spyder/blob/master/CONTRIBUTING.md
TEST_SPYDER = "pytest -n0 -rA"
SPECS_SPYDER = {
    k: {
        "python": "3.9", 
        "packages": "environment.yml",
        "pre_install": [
            "conda env update --file requirements/linux.yml",
            "conda env update --file requirements/tests.yml"
        ],
        "install": "python -m pip install -e .;",
        # "install": "python -m pip install 'numpy<2.0'; python -m pip install --upgrade Cython; python -m pip install -r requirements-dev.txt; python -m pip install -e .",
        "test_cmd": TEST_SPYDER,
    }
    for k in []
}

# hypothesis
# https://github.com/HypothesisWorks/hypothesis/blob/eaafdfcad3f362e75746863472101d4cfabbc33d/CONTRIBUTING.rst
TEST_HYPOTHESIS = "pytest -n0 -rA --tb=no --no-header"
SPECS_HYPOTHESIS = {
    k: {
        "python": "3.10", 
        "packages": "requirements.txt", # this installs tools.txt
        "install": "python -m pip install -r requirements/test.txt; python -m pip install -e hypothesis-python/;",
        "test_cmd": TEST_HYPOTHESIS,
    }
    for k in ['3.55', '3.61', '3.60', '3.59', '3.63', '3.66', '3.67', '3.68', '3.69', '3.70', '5.1', '5.5', '5.24', '5.6', '5.9', '5.8', '5.10', '5.12', '5.15', '5.20', '5.23', '5.36', '5.32', '5.33', '5.38', '5.41', '5.42', '5.43', '5.47', '6.1', '6.4', '6.6', '6.8', '6.14', '6.13', '6.18', '6.21', '6.24', '6.28', '6.29', '3.73', '3.71', '3.75', '3.79', '3.82', '3.85', '3.88', '4.0', '3.86', '4.2', '4.4', '4.15', '4.12', '4.14', '4.18', '4.23', '4.24', '4.26', '4.32', '4.38', '4.40', '4.42', '4.46', '4.44', '4.50', '4.54', '4.55', '5.2', '5.4', '6.30', '6.31', '6.36', '6.40', '6.43', '6.53', '6.45', '6.46', '6.47', '6.50', '6.54', '6.59', '6.62', '6.66', '6.71', '6.74', '6.77', '6.81', '6.87', '6.88', '6.93', '6.98', '6.99', '6.100', '6.102']
}
for k in ['3.55', '3.61', '3.60', '3.59', '3.63', '3.66', '3.67', '3.68', '3.69', '3.70', '5.1', '5.5', '5.24', '5.6', '5.9', '5.8', '5.10', '5.12', '5.15', '5.20', '5.23', '5.36', '5.32', '5.33', '5.38', '5.41', '5.42', '5.43', '5.47', '6.1', '6.4', '6.6', '6.8', '6.14', '6.13', '6.18', '6.21', '6.24', '6.28', '6.29', '3.73', '3.71', '3.75', '3.79', '3.82', '3.85', '3.88', '4.0', '3.86', '4.2', '4.4', '4.15', '4.12', '4.14', '4.18', '4.23', '4.24', '4.26', '4.32', '4.38', '4.40', '4.42', '4.46', '4.44', '4.50', '4.54', '4.55', '5.2', '5.4', '6.30', '6.31']:
    SPECS_HYPOTHESIS[k]['python'] = '3.9'

# pydantic
# https://docs.pydantic.dev/latest/contributing/
# TEST_PYDANTIC = 'export PATH="$HOME/.local/bin:$PATH"; pdm run coverage run -m pytest -rA --tb=short --no-header'
TEST_PYDANTIC = 'pytest -rA --tb=short -vv -o console_output_style=classic --no-header'
SPECS_PYDANTIC = {
    k: {
        "python": "3.8",
        "pre_install": [
            "apt-get update && apt-get install -y locales",
            "apt-get install -y pipx",
            "pipx ensurepath",
            # well, this in fact uses python 3.10 as default by pipx
            "pipx install pdm",
            'export PATH="$HOME/.local/bin:$PATH"',
            "which python",
            "python --version",
        ],
        "install": 'export PATH="$HOME/.local/bin:$PATH"; pdm add pre-commit; make install;',
        "test_cmd": TEST_PYDANTIC,
    }
    for k in ['0.2', '0.41', '0.4', '0.6', '0.9', '0.10', '0.11', '0.13', '0.14', '0.151', '0.15', '0.17', '0.18', '0.201', '0.20', '0.24', '0.27', '0.29', '1.01', '0.32', '1.4', '1.31', '1.41', '1.51', '1.5', '1.71', '1.6', '1.7', '1.8', '1.9', '1.10', '2.0', '2.01', '2.02', '2.03', '2.04', '2.6', '2.5', '2.4', '2.7']
}

for k in ['0.2', '0.41', '0.4', '0.6', '0.9', '0.10', '0.11', '0.13', '0.14', '0.151', '0.15', '0.17', '0.18', '0.201', '0.20', '0.24', '0.27', '0.29', '1.01', '0.32', '1.4', '1.31', '1.41', '1.51', '1.5', '1.71', '1.6', '1.7', '1.8', '1.9', '1.10']:
    # not working yet
    SPECS_PYDANTIC[k]["pre_install"] = [
            "apt-get update && apt-get install -y locales",
            "apt-get install -y pipx",
            "pipx ensurepath",
            # well, this in fact uses python 3.10 as default by pipx
            "pipx install pdm  --python python3.7",
            'export PATH="$HOME/.local/bin:$PATH"',
            "which python",
            "python --version",
        ]
    SPECS_PYDANTIC[k]["python"] = "3.7"

# pandas
# https://pandas.pydata.org/pandas-docs/dev/development/contributing_environment.html
TEST_PANDAS = "pytest -rA --tb=long"
SPECS_PANDAS = {
    k: {
        "packages": "environment.yml",
        "pre_install": [
            "git remote add upstream https://github.com/pandas-dev/pandas.git",
            "git fetch upstream --tags"
        ],
        "install": "python -m pip install -ve . --no-build-isolation -Ceditable-verbose=true; pip uninstall pytest-qt -y;",
        "test_cmd": TEST_PANDAS,
    }
    for k in ['0.16', '0.17', '0.18', '0.19', '0.20', '0.21', '0.22', '0.23', '0.24', '0.25', '0.26', '1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '2.0', '2.1', '2.2', '3.0']
}
for k in ['0.16', '0.17', '0.18', '0.19', '0.20', '0.21', '0.22', '0.23', '0.24', '0.25', '0.26', '1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '2.0', '2.1']:
    # numpy 2 is supported in pandas 2.2
    SPECS_PANDAS[k]['install'] = "python -m pip install 'numpy<2'; " + SPECS_PANDAS[k]['install'] 

# hydra
TEST_HYDRA = "pytest -rA --tb=long"
SPECS_HYDRA = {
    k: {
        "python": "3.8",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y openjdk-17-jdk openjdk-17-jre",
        ],
        "install": "pip install -r requirements/dev.txt; pip install -e .;",
        "test_cmd": TEST_HYDRA,
    }
    for k in ['0.1', '0.9', '0.10', '0.11', '0.12', '1.0', '1.1', '1.2', '1.3', '1.4']
}
for k in ['0.1', '0.9', '0.10', '0.11', '0.12', '1.0', '1.1', '1.2']:
    # fix omegaconf pip version issue
    SPECS_HYDRA[k]['install'] = '{ tail -n1 requirements/requirements.txt | grep -q "." && echo ""; } >> requirements/requirements.txt; echo "pip==24.0" >> requirements/requirements.txt;' + 'pip install "pip==24.0"; ' + SPECS_HYDRA[k]['install']
    # isort is moved to PyCQA now
    SPECS_HYDRA[k]['install'] = "sed -i 's|isort@git+git://github.com/timothycrosley/isort|isort@git+https://github.com/timothycrosley/isort|g' requirements/dev.txt; " + SPECS_HYDRA[k]['install']

# --------------------------------------------------------------------------- #
# SWE-Gym
# --------------------------------------------------------------------------- #

# Constants - Task Instance Instllation Environment
MAP_REPO_VERSION_TO_SPECS_PY = {
    "astropy/astropy": SPECS_ASTROPY,
    "dbt-labs/dbt-core": SPECS_DBT_CORE,
    "django/django": SPECS_DJANGO,
    "matplotlib/matplotlib": SPECS_MATPLOTLIB,
    "marshmallow-code/marshmallow": SPECS_MARSHMALLOW,
    "mwaskom/seaborn": SPECS_SEABORN,
    "pallets/flask": SPECS_FLASK,
    "psf/requests": SPECS_REQUESTS,
    "pvlib/pvlib-python": SPECS_PVLIB,
    "pydata/xarray": SPECS_XARRAY,
    "pydicom/pydicom": SPECS_PYDICOM,
    "pylint-dev/astroid": SPECS_ASTROID,
    "pylint-dev/pylint": SPECS_PYLINT,
    "pytest-dev/pytest": SPECS_PYTEST,
    "pyvista/pyvista": SPECS_PYVISTA,
    "scikit-learn/scikit-learn": SPECS_SKLEARN,
    "sphinx-doc/sphinx": SPECS_SPHINX,
    "sqlfluff/sqlfluff": SPECS_SQLFLUFF,
    "swe-bench/humaneval": SPECS_HUMANEVAL,
    "sympy/sympy": SPECS_SYMPY,
    # --------------------------------------------------------------------------- #
    # SWE-Gym
    # --------------------------------------------------------------------------- #
    "python/mypy": SPECS_MYPY,
    "getmoto/moto": SPECS_MOTO,
    "conan-io/conan": SPECS_CONAN,
    "dask/dask": SPECS_DASK,
    "project-monai/monai": SPECS_MONAI,
    "iterative/dvc": SPECS_DVC,
    "bokeh/bokeh": SPECS_BOKEH,
    "modin-project/modin": SPECS_MODIN,
    "spyder-ide/spyder": SPECS_SPYDER,
    "HypothesisWorks/hypothesis": SPECS_HYPOTHESIS,
    "pydantic/pydantic": SPECS_PYDANTIC,
    "pandas-dev/pandas": SPECS_PANDAS,
    "facebookresearch/hydra": SPECS_HYDRA,
}

# Constants - Repository Specific Installation Instructions
MAP_REPO_TO_INSTALL_PY = {}


# Constants - Task Instance Requirements File Paths
MAP_REPO_TO_REQS_PATHS = {
    "dbt-labs/dbt-core": ["dev-requirements.txt", "dev_requirements.txt"],
    "django/django": ["tests/requirements/py3.txt"],
    "matplotlib/matplotlib": [
        "requirements/dev/dev-requirements.txt",
        "requirements/testing/travis_all.txt",
    ],
    "pallets/flask": ["requirements/dev.txt"],
    "pylint-dev/pylint": ["requirements_test.txt"],
    "pyvista/pyvista": ["requirements_test.txt", "requirements.txt"],
    "sqlfluff/sqlfluff": ["requirements_dev.txt"],
    "sympy/sympy": ["requirements-dev.txt", "requirements-test.txt"],
}

# --------------------------------------------------------------------------- #
# Not used
# --------------------------------------------------------------------------- #

# Constants - Task Instance environment.yml File Paths
MAP_REPO_TO_REQS_PATHS = {
    "dbt-labs/dbt-core": ["dev-requirements.txt", "dev_requirements.txt"],
    "django/django": ["tests/requirements/py3.txt"],
    "matplotlib/matplotlib": [
        "requirements/dev/dev-requirements.txt",
        "requirements/testing/travis_all.txt",
    ],
    "pallets/flask": ["requirements/dev.txt"],
    "pylint-dev/pylint": ["requirements_test.txt"],
    "pyvista/pyvista": ["requirements_test.txt", "requirements.txt"],
    "sqlfluff/sqlfluff": ["requirements_dev.txt"],
    "sympy/sympy": ["requirements-dev.txt"],
    "Project-MONAI/MONAI": ["requirements-dev.txt"],
    "HypothesisWorks/hypothesis": ["requirements/tools.txt"],
    "facebookresearch/hydra": ['requirements/dev.txt']
}

# Constants - Task Instance environment.yml File Paths
MAP_REPO_TO_ENV_YML_PATHS = {
    "matplotlib/matplotlib": ["environment.yml"],
    "pydata/xarray": ["ci/requirements/environment.yml", "environment.yml"],
    "bokeh/bokeh": [
        # for v3
        "conda/environment-test-3.10.yml",
        #for v2
        "environment.yml"
        # for v1
        ],
    "modin-project/modin": [
        "environment-dev.yml"
        ],
    "dask/dask": [
        "continuous_integration/environment-3.10.yaml",
        "continuous_integration/environment-3.9.yaml",
        "continuous_integration/environment-3.8.yaml",
        "continuous_integration/travis/travis-37.yaml"
    ],
    "spyder-ide/spyder": [
        "requirements/main.yml",
    ],
    "pandas-dev/pandas": [
        "environment.yml"
    ]
}
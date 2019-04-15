from setuptools import setup, find_packages
import sys

VERSION = {}
with open("csqa/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

# make pytest-runner a conditional requirement,
# per: https://github.com/pytest-dev/pytest-runner#considerations
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

setup_requirements = [
    # add other setup requirements as necessary
] + pytest_runner

setup(name='csqa',
      version=VERSION["VERSION"],
      description='An library for commonsense qa, built on allennlp.',
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      classifiers=[
          'Intended Audience :: Science/Research',
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      keywords='commensenseqa NLP deep learning machine reading',
      url='https://github.com/Shuailong/csqa',
      author='Liang Shuailong',
      author_email='liangshuailong@gmail.com',
      license='Apache',
      packages=find_packages(exclude=["*.tests", "*.tests.*",
                                      "tests.*", "tests"]),
      install_requires=[
          'allennlp>=0.8.3',
          'pytest',
          'pytest-pythonpath'
      ],
      setup_requires=setup_requirements,
      tests_require=[
          'pytest',
          'flaky',
          'responses>=0.7',
          'moto==1.3.4',
      ],
      include_package_data=True,
      python_requires='>=3.6.8',
      zip_safe=False)

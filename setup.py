"""
Setup script for DEvolve package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name='devolve',
    version='1.0.0',
    author='DEvolve Development Team',
    author_email='your.email@example.com',
    description='Comprehensive Differential Evolution library for Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/devolve',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'matplotlib>=3.5.0',
        'pandas>=1.3.0',
        'seaborn>=0.11.0',
        'tqdm>=4.62.0',
        'scikit-learn>=1.0.0',
    ],
    extras_require={
        'gpu': ['cupy>=10.0.0'],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx_rtd_theme>=1.0.0',
            'sphinx-autodoc-typehints>=1.12.0',
        ],
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
            'isort>=5.10.0',
        ],
        'all': [
            'cupy>=10.0.0',
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'sphinx>=4.0.0',
            'sphinx_rtd_theme>=1.0.0',
        ],
    },
    keywords='differential-evolution optimization genetic-algorithm metaheuristic',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/devolve/issues',
        'Source': 'https://github.com/yourusername/devolve',
        'Documentation': 'https://devolve.readthedocs.io',
    },
)

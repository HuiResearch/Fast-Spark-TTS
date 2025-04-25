# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/4/25 12:43
# Author  : Hui Huang

import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))


def get_requires() -> list[str]:
    with open("requirements.txt", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines


def get_readme() -> str:
    with open(os.path.join(here, 'README.MD'), encoding='utf-8') as f:
        long_description = f.read()
    return long_description


setup(
    name='flashtts',
    version='0.1.0',
    description='A Fast TTS toolkit',
    long_description=get_readme(),
    long_description_content_type='text/markdown',
    keywords=["flashtts", "tts", "sparktts", "spark-tts", "megatts3", "orpheus-tts", "vllm", "sglang", "llama-cpp"],
    author='HuangHui',
    author_email='m13021933043@163.com',
    url='https://github.com/HuiResearch/FlashTTS',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        # 把 templates 目录下的静态文件也一起打包
        'flashtts.server.templates': ['*.html', '*.ico'],
    },
    install_requires=get_requires(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    entry_points={"console_scripts": ["flashtts = flashtts.commands.flashtts_cli:main"]},
)

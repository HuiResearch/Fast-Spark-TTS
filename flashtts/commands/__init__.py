# -*- coding: utf-8 -*-
# Project : FlashTTS
# Time    : 2025/4/25 11:13
# Author  : Hui Huang
from abc import ABC, abstractmethod
from argparse import ArgumentParser


class BaseCLICommand(ABC):
    @staticmethod
    @abstractmethod
    def register_subcommand(parser: ArgumentParser):
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()

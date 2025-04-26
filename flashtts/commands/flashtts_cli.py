# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/4/25 11:24
# Author  : Hui Huang
from argparse import ArgumentParser

from flashtts.commands.serve import ServerCommand
from flashtts.commands.infer import InferCommand


def main():
    parser = ArgumentParser(
        "FlashTTS CLI tool",
        usage="flashtts <command> [<args>]",
        epilog="For more information about a command, run: `flashtts <command> --help`"
    )
    commands_parser = parser.add_subparsers(help="flashtts command helpers")

    # Register commands
    ServerCommand.register_subcommand(commands_parser)
    InferCommand.register_subcommand(commands_parser)

    # Let's go
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    service = args.func(args)
    service.run()


if __name__ == "__main__":
    main()

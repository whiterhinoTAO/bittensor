# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

import os
import argparse
import bittensor
from rich.tree import Tree
from .utils import get_coldkey_wallets_for_path, get_hotkey_wallets_for_wallet

class ListCommand:
    @staticmethod
    def run (cli):
        r""" Lists wallets."""
        all_coldkey_wallets = get_coldkey_wallets_for_path( cli.config.wallet.path )
        if len(all_coldkey_wallets) == 0:
            bittensor.__console__.print("[bold red]No wallets found.")
            return

        root = Tree("Wallets")
        for coldkey_wallet in all_coldkey_wallets:
            w_name = coldkey_wallet.name
            try:
                if coldkey_wallet.coldkeypub_file.exists_on_device() and not coldkey_wallet.coldkeypub_file.is_encrypted():
                    coldkeypub_str = coldkey_wallet.coldkeypub.ss58_address
                else:
                    coldkeypub_str = '?'
            except:
                coldkeypub_str = '?'

            wallet_tree = root.add("\n[bold white]{} ({})".format(w_name, coldkeypub_str))

            hotkeys = get_hotkey_wallets_for_wallet(coldkey_wallet)
            if len( hotkeys ) > 0:
                for hotkey_wallet in hotkeys:
                    h_name = hotkey_wallet.hotkey_str
                    try:
                        if hotkey_wallet.hotkey_file.exists_on_device() and not hotkey_wallet.hotkey_file.is_encrypted():
                            hotkey_str = hotkey_wallet.hotkey.ss58_address
                        else:
                            hotkey_str = '?'
                    except:
                        hotkey_str = '?'
                    wallet_tree.add("[bold grey]{} ({})".format(h_name, hotkey_str))

        if len(all_coldkey_wallets) == 0:
            root.add("[bold red]No wallets found.")

        # Uses rich print to display the tree.
        bittensor.__console__.print(root)

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        pass

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        list_parser = parser.add_parser(
            'list', 
            help='''List wallets'''
        )
        list_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        list_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )
        bittensor.wallet.add_args( list_parser )
        bittensor.subtensor.add_args( list_parser )
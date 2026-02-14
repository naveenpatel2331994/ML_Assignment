#!/usr/bin/env python3
import argparse
import subprocess
import sys

SCRIPTS = {
    'download': 'download_ucibank.py',
    'eda': 'eda_bank.py',
    'train': 'train_baseline.py'
}

def run(script_name):
    cmd = [sys.executable, script_name]
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd)

def main():
    p = argparse.ArgumentParser(description='Project CLI: download, eda, train')
    p.add_argument('command', choices=['download','eda','train','all','deps'], help='command to run')
    args = p.parse_args()

    if args.command == 'deps':
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', '-r', 'requirements.txt'])
        return

    if args.command == 'all':
        run(SCRIPTS['download'])
        run(SCRIPTS['eda'])
        run(SCRIPTS['train'])
        return

    run(SCRIPTS[args.command])

if __name__ == '__main__':
    main()

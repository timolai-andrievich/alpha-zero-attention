#!python3
import argparse
import os
import sys

import tqdm
import yaml

from connect4_mcts import coach
from connect4_mcts import policy


DEFAULT_CONFIG = {
    'filters': 128,
    'blocks': 10,
    'buffer_size': 20000,
    'epochs': 10,
    'iterations': 3,
    'learning_rate': 2e-4,
    'playout': 80,
    'episodes_per_iteration': 25,
    'batch_size': 256,
    'minibatch_size': 32,
    'c_puct': 5.0,
    'temp': 0,
    'device': 'cpu',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-o', '--output-model', default='model.pt')
    parser.add_argument('-m', '--model', required=False)
    args = parser.parse_args()
    config = DEFAULT_CONFIG.copy()
    if args.config_file is not None:
        try:
            with open(args.config_file) as config_file:
                loaded_config = yaml.safe_load(config_file)
                config.update(loaded_config)
        except Exception as e:
            print(f'Exception while reading the config file: {e}', file=sys.stderr)
            sys.exit(1)
    model = policy.Model(
        config['filters'], config['blocks'], config['learning_rate'], config['device'])
    if args.model is not None:
        try:
            model.load(args.model)
        except Exception as e:
            print(f'Exception while loading network weights: {e}', file=sys.stderr)
            sys.exit(1)
    c = coach.Coach(model, config['buffer_size'])
    pbar = tqdm.trange(config['iterations']) if args.verbose else range(config['iterations'])
    for _ in pbar:
        c.generate_games(config['episodes_per_iteration'],
                         config['c_puct'], config['playout'], config['temp'])
        loss = c.train_epochs(config['batch_size'],
                       config['minibatch_size'], config['epochs'])
        model.save(args.output_model)
        pbar.set_description(f'loss: {loss:.4f}')
    model.save(args.output_model)


if __name__ == '__main__':
    main()

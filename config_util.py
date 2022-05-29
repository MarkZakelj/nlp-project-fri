import os
import toml


def check_train_config(configs):
    """check if config values are consistent"""
    must_have_keys = ['experiment', 'model_id', 'max_length', 'batch_size', 'epochs']
    experiments = {}
    for conf in configs:
        for key in must_have_keys:
            if key not in conf:
                raise KeyError(f'Missing key in the config dictionary: {key} not found in  {conf}')
        if conf['experiment'] not in experiments:
            experiments[conf['experiment']] = set()
        if conf['model_id'] in experiments[conf['experiment']]:
            raise NameError(
                f'This model is already a part of an experiment: {conf["model_id"]} already in {conf["experiment"]}')
        experiments[conf['experiment']].add(conf['model_id'])
    return True


def get_data_config():
    data_config = toml.load(open('config_files/config_data.toml'))
    # for each dataset configuration, create new normalized configuration
    # such that keys "train" and "test" are keys in one configuration
    # and one config doesnt include sub-lists
    normalized_config = []
    for conf in data_config['data']:
        if 'datasets' not in conf.keys():
            for train_test in data_config['default_datasets']:
                normalized_config.append(dict(conf, **train_test))
    return normalized_config


def get_data_reg_config():
    data_config = toml.load(open('config_files/config_data_reg.toml'))
    return data_config['data']


def get_train_config():
    train_config = toml.load(open('config_files/config_train.toml'))
    default_values = train_config['default_values']
    train = train_config['train']
    normalized_config = []
    # create normalized configs without sublists
    for i, experiment in enumerate(train):
        for j, model in enumerate(experiment['model']):
            # update missing values with default values
            norm_conf = dict(default_values, **train[i]['model'][j])
            normalized_config.append(dict({'experiment': experiment['experiment']}, **norm_conf))
    check_train_config(normalized_config)
    return normalized_config


def get_train_reg_config():
    train_config = toml.load(open('config_files/config_train_reg.toml'))['train']
    check_train_config(train_config)
    return train_config


def list_experiments():
    exp = sorted(os.listdir(os.path.join('data', 'experiments')))
    return exp


def list_models(experiment):
    mods = sorted(os.listdir(os.path.join('data', 'experiments', experiment)))
    mods = [m for m in mods if os.path.isdir(os.path.join('data', 'experiments', experiment, m))]
    return mods


def main():
    # testing
    import json
    conf = get_train_reg_config()
    print(json.dumps(conf, indent=4))


if __name__ == '__main__':
    main()

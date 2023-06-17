import sys
from ast import literal_eval

from nano_gpt.Config.config import CONFIG_MAP


def registerConfig(config_name):
    valid_config_name_list = list(CONFIG_MAP.keys())
    if config_name not in valid_config_name_list:
        print('[WARN][configurator::registerConfig]')
        print('\t config_name not valid! these are valid names:')
        print('\t', valid_config_name_list)
        return None

    config_dict = CONFIG_MAP[config_name]

    print(f"Overriding config with {config_name}:")

    for arg in sys.argv[1:]:
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]

        try:
            # attempt to eval it it (e.g. if bool, number, or etc)
            attempt = literal_eval(val)
        except (SyntaxError, ValueError):
            # if that goes wrong, just use the string
            attempt = val
        # ensure the types match ok
        assert isinstance(attempt, globals()[key])
        # cross fingers
        print(f"Overriding: {key} = {attempt}")
        config_dict[key] = attempt
    return config_dict

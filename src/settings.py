from collections import namedtuple

import ruamel.yaml as yaml
import os

def dict_to_namedtuple(dic: dict):
    return namedtuple('tuple', dic.keys())(**dic)


class Settings:
    def __init__(self):
        pass

    def load_settings(self, dataset):
        settings_file = './model_settings/'+dataset+'.yaml'
        with open(settings_file, 'r') as f:
            setting = yaml.load(f, Loader=yaml.RoundTripLoader)
        self.data = dict_to_namedtuple(setting['data'])
        self.model = dict_to_namedtuple(setting['model'])
        self.trainer = dict_to_namedtuple(setting['trainer'])

        self.data_dict = setting['data']
        self.model_dict = setting['model']
        self.trainer_dict = setting['trainer']


def load_server_config():

    settings_file = '../server_config.yaml'
    with open(settings_file, 'r') as f:
        setting = yaml.load(f, Loader=yaml.RoundTripLoader)
    server_config = dict_to_namedtuple(setting['config'])

    return server_config
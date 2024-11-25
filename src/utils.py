import yaml

class UtilFunction():
    @staticmethod
    def load_config(config_path:str):
        with open(config_path, 'rb') as f:
            return yaml.safe_load(f)
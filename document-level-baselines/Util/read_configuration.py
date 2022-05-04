import yaml
from ray import tune
####### Read YML Model Hyper-parameter Configuration YML file

def flat_dict(config_dict):
    def remove_grid_search(input_dict):
        parameter_config = {}
        for key, value in input_dict.items():
            if key == "grid_search":
                continue
            elif type(value) is dict:
                parameter_config = {**remove_grid_search(value), **parameter_config}
            else:
                parameter_config[key] = value
        return parameter_config

    config = remove_grid_search(config_dict)
    return config

def hyper_parameter_search(config_dict):
    def locate_grid_search(input_dict):
        parameter_config = None
        for key, value in input_dict.items():
            if key == "grid_search":
                parameter_config = {
                    hyper: tune.grid_search(value)
                    for hyper, value in value.items()
                }
                return parameter_config
            elif key == "bayesian":
                parameter_config = {

                }
            elif type(value) is dict:
                parameter_config = locate_grid_search(value)
        return parameter_config

    config = locate_grid_search(config_dict)
    search_hyper = list(config.keys())

    return config, search_hyper

def extract_yaml(config_path):
    def check_yaml_file(doc):
        new_doc = {}
        for key, value in doc.items():
            if type(value) is dict:
                new_value = check_yaml_file(value)
            elif type(value) is str and ".yml" in value:
                new_value = extract_yaml(value)
            else:
                new_value = value
            new_doc[key] = new_value
        return new_doc

    with open(config_path) as f1:
        docs = yaml.load_all(f1, Loader=yaml.FullLoader)
        doc = next(docs)
        doc = check_yaml_file(doc)
    return doc

def read_yaml(config_path):
    doc = extract_yaml(config_path)
    tune_config, search_parameters = hyper_parameter_search(doc)
    general_config = flat_dict(doc)
    return tune_config, search_parameters, general_config
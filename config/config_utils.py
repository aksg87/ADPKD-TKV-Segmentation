from importlib import import_module

"""Config parsing utilities"""


def get_simple_instance(module_name, class_name, processed_params):
    """
    Returns a class instance.

    Args:
        module_name: str, module name
        class_name: str, class_name
        processed_params: dict, class params
    """
    module = import_module(module_name)
    class_ = getattr(module, class_name)
    instance = class_(**processed_params)

    return instance


def is_object_config(object_config):
    return isinstance(object_config, dict) and "_CLASS_INFO" in object_config


def get_object_instance(object_config):
    """
    Return class instance from config.

    Args:
        object_config, dict including the special "_CLASS_INFO"

    Example:
        object_config = {"key_1": object_1_config,
                         "key_2": simple_param_2,
                         "key_3": object_3_config,
                         "_CLASS_INFO": {"_MODULE_NAME": loss_utils.losses,
                                         "_CLASS_NAME": BaselineLoss}
        }
        `object_1_config` and `object_3_config` are also dicts of the same type
        as `object_config` (must include "_CLASS_INFO")
    """
    processed_params = {}

    for key, value in object_config.items():
        #  special info key is not to be used inside `get_simple_instance`
        if key == "_CLASS_INFO":
            continue
        # nested object
        if isinstance(value, dict):
            processed_params[key] = get_object_instance(value)
        # value is a leaf
        else:
            processed_params[key] = value

    if is_object_config(object_config):
        class_info = object_config["_CLASS_INFO"]
        module_name = class_info["_MODULE_NAME"]
        class_name = class_info["_CLASS_NAME"]
        return get_simple_instance(module_name, class_name, processed_params)

    return processed_params

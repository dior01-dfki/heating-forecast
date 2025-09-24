from forcateri.utils.config_utils import extract_config, from_args_to_kwargs, load_config
import argparse
import yaml

def arg_parser(project_root):
    parser = argparse.ArgumentParser()
    #project_root=Path(__file__).parent.parent
    config_path = project_root.joinpath("configs")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Configuration file name without .yaml extension"
    )
    args,remaining_args = parser.parse_known_args()
    with open(config_path.joinpath(args.config + '.yaml'),"r") as infile:
            parsed_config = yaml.safe_load(infile)
    args = extract_config(parsed_config)
    for k, v in args:
        if isinstance(v, list):
            v = ",".join(map(str, v))
        elif v is None:
            v = "None"
        parser.add_argument(f"--{k}", default=v)
    return parser
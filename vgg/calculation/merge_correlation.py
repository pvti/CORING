import logging
import argparse
import json


def get_parser():
    parser = argparse.ArgumentParser(description='Merge calculated correlations'
                                     )
    parser.add_argument('--input',
                        default=[
                            'calculation/baseline/vgg/correlation_full_vgg16.json',
                            'calculation/baseline/vgg/correlation_decomposition_vgg16.json'
                        ],
                        type=str,
                        nargs='+',
                        help='path to precalculated calculation',
                        )
    parser.add_argument('--output',
                        default='calculation/baseline/vgg/correlation_vgg16.json',
                        help='path to save calculation',
                        )
    parser.add_argument('--log_file',
                        default='logs/merge_correlation_vgg16.txt',
                        help='path to log file',
                        )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(args.log_file),
                                  logging.StreamHandler()
                                  ]
                        )
    logging.info("Arguments: " + str(args))

    # load pre-calculated correlations
    logging.info(f"=> loading '{args.input[0]}'")
    f = open(args.input[0], 'r+')
    data_full = json.loads(f.read())
    dict = data_full
    dict["input"] = args.input

    for i in range(1, len(args.input)):
        logging.info(f"=> loading '{args.input[i]}'")
        f = open(args.input[i], 'r+')
        data = json.loads(f.read())
        for decomposer, decomposer_dict in data["correlation"].items():
            dict["correlation"][decomposer] = decomposer_dict

    with open(args.output, 'w') as file:
        logging.info(f"=> dumping to {args.output}")
        json.dump(dict, file, indent=4)

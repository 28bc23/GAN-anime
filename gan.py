import argparse
import sys

parser = argparse.ArgumentParser(
    description='gan.py allows you to easily manage GANs in this project for easy generation and training.'
)
subparsers = parser.add_subparsers(dest="command")

install_parser = subparsers.add_parser("install", help="install environment for GANs (Only for Linux)")
generate_parser = subparsers.add_parser("generate", help="set gan to generation mode")
train_parser = subparsers.add_parser("train", help="set gan to train mode")

'''
Generation
'''
generate_parser.add_argument("-m", "--model", choices=['progan', 'gan'], help="select model for generating: [progan/gan]", default="progan")
generate_parser.add_argument("-o", "--output", help="Specify name of output image", default="gen")

'''
Train
'''
train_parser.add_argument("-m", "--model", choices=['progan', 'gan'], default="progan", help="select model for training: [progan/gan]")
train_parser.add_argument("--cpu", action="store_true", default=False, help="Will train on CPU")
train_parser.add_argument("-s", "--silent", action="store_true", default=False, help="Suppress logging to terminal")
train_parser.add_argument("-c", "--clean", action="store_true", default=False, help="Train model without loading last parameters")
train_parser.add_argument("-e", "--epochs", type=int, default=10, help="Set train epochs")
train_parser.add_argument("-b", "--batch", type=int, default=128, help="Set batch size")
train_parser.add_argument('-dlr', "--discriminatorLearningRate", type=float, default=1e-4, help="Set discriminators learning rate")
train_parser.add_argument('-glr', "--generatorLearningRate", type=float, default=1e-4, help="Set generators learning rate")
train_parser.add_argument('-gi', "--generatorItters", type=int, default=1, help="Set generators itters (how many times will generator optimalize per step)")
train_parser.add_argument('-di', "--discriminatorsItters", type=int, default=1, help="Set discriminators itters (how many times will discriminator optimalize per step)")


args = parser.parse_args()

if args.command is None:
    parser.print_help()
    sys.exit(1)

print("Použitý subparser:", args.command)
print(args)

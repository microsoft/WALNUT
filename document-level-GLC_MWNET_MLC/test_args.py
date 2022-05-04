import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--my_variable', default=False, action='store_true')
parser.add_argument('--other_variable', default=True, action='store_false')
args = parser.parse_args()

print(args)

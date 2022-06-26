import argparse


def parse_args():
    parser = argparse.ArgumentParser(prog='Leapfrog PY Join')
    parser.add_argument('-query', help='path to query file')
    parser.add_argument('-tables', help='paths to tabels separatred by commas')

    args = parser.parse_args()
    return args

def leapfrog_triejoin_main():
    args = parse_args()
    pass

if __name__ == '__main__':
    leapfrog_triejoin_main()
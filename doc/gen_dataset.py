import argparse
import os

DATASET = ["Cance", "France", "Lez"]
DATASET_PATH = f"{os.path.dirname(os.path.realpath(__file__))}/../smash/factory/dataset"

# TODO: Refactorize this when it gets more complicated. Pass to each dataset the files or directories
# that we want to be uploaded and used in the documentation.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "-dataset",
        "--dataset",
        nargs="+",
        default=DATASET,
        choices=DATASET,
        help="dataset to generate",
    )

    parser.add_argument(
        "-t",
        "-tar",
        "--tar",
        default=False,
        action="store_true",
        help="Skip tests to include",
    )

    args = parser.parse_args()

    for ds in args.dataset:
        ds_dir = f"{ds}-dataset"
        if os.path.exists(ds_dir):
            continue
        os.system(f"mkdir -p {ds_dir}")
        os.system(f"cp {DATASET_PATH}/France_flwdir.tif {ds_dir}/.")
        os.system(f"cp -r {DATASET_PATH}/{ds}/pet {DATASET_PATH}/{ds}/prcp {ds_dir}/.")

        if ds == "Cance":
            os.system(f"cp -r {DATASET_PATH}/{ds}/qobs {DATASET_PATH}/{ds}/gauge_attributes.csv {ds_dir}/.")

        elif ds == "Lez":
            os.system(
                f"cp -r {DATASET_PATH}/{ds}/qobs {DATASET_PATH}/{ds}/gauge_attributes.csv "
                f"{DATASET_PATH}/{ds}/descriptor {ds_dir}/."
            )

        if args.tar:
            os.system(f"tar cf {ds_dir}.tar {ds_dir}")
            os.system(f"rm -r {ds_dir}")

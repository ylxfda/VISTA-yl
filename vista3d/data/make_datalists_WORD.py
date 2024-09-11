# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Step 1.
# reading image and label folders, listing all the nii.gz files,
# creating a data_list.json for training and validation
# the data_list.json format is like ('testing' labels are optional):
# {
#     "training": [
#         {"image": "img0001.nii.gz", "label": "label0001.nii.gz", "fold": 0},
#         {"image": "img0002.nii.gz", "label": "label0002.nii.gz", "fold": 2},
#         ...
#     ],
#     "testing": [
#         {"image": "img0003.nii.gz", "label": "label0003.nii.gz"},
#         {"image": "img0004.nii.gz", "label": "label0004.nii.gz"},
#         ...
#     ]
# }

import os
import re
from glob import glob
from pprint import pprint

from monai.apps import get_logger
from monai.bundle import ConfigParser
from monai.data.utils import partition_dataset

logger = get_logger(__name__)
test_ratio = 0.2  # test split
n_folds = 5  # training and validation split
seed = 20230808  # random seed for data partition_dataset reproducibility
output_json_dir = os.path.join(os.path.dirname(__file__), "jsons")


def register_make(func):
    """
    register the function to make the data list
    """
    global _make_funcs
    if "_make_funcs" not in globals():
        _make_funcs = {}
    if func.__name__ in _make_funcs:
        raise ValueError(f"{func.__name__} already exists.")
    _make_funcs[func.__name__] = func
    return func


def search_image_files(base_dir, ext, regex=None):
    """returns a list of relative filenames with given extension in `base_dir`"""
    print(f"searching ext={ext} from base_dir={base_dir}")
    images = []
    for root, _, files in os.walk(base_dir):
        images.extend(
            os.path.join(root, filename) for filename in files if filename.endswith(ext)
        )
    if regex is not None:
        images = [x for x in images if re.compile(regex).search(x) is not None]
    print(f"found {len(images)} *.{ext} files")
    return sorted(images)


def create_splits_and_write_json(
    images,
    labels,
    ratio,
    num_folds,
    json_name,
    rng_seed,
    label_dict,
    original_label_dict=None,
):
    """
    first generate training/test split, then from the training part,
    generate training/validation num_folds
    """
    items = [{"image": img, "label": lab} for img, lab in zip(images, labels)]
    train_test = partition_dataset(
        items, ratios=[1 - ratio, ratio], shuffle=True, seed=rng_seed
    )
    print(f"training: {len(train_test[0])}, testing: {len(train_test[1])}")
    train_val = partition_dataset(
        train_test[0], num_partitions=num_folds, shuffle=True, seed=rng_seed
    )
    print(f"training validation folds sizes: {[len(x) for x in train_val]}")
    training = []
    for f, x in enumerate(train_val):
        for item in x:
            item["fold"] = f
            training.append(item)

    # write json
    parser = ConfigParser({})
    parser["training"] = training
    parser["testing"] = train_test[1]

    parser["label_dict"] = label_dict
    parser["original_label_dict"] = original_label_dict or label_dict

    print(f"writing {json_name}\n\n")
    if os.path.exists(json_name):
        logger.warning(f"rewrite existing datalist file: {json_name}")
    ConfigParser.export_config_file(parser.config, json_name, indent=4)


def filtering_files(base_url, image_names, label_names, idx=-1):
    """
    check the idx-th item in the lists of image and label filenames, remove:

        - image files without corresponding label files

    """
    _tmp_img = os.path.join(base_url, image_names[idx])
    _tmp_lab = os.path.join(base_url, label_names[idx])
    if not (os.path.exists(_tmp_img) and os.path.exists(_tmp_lab)):
        if not os.path.exists(_tmp_img):
            logger.warning(f"image file {_tmp_img} pair does not exist")
        if not os.path.exists(_tmp_lab):
            logger.warning(f"label file {_tmp_lab} pair does not exist")
        image_names.pop(idx)
        label_names.pop(idx)
        
        
####
@register_make
def make_word():
    base_url = "/data/DataSSD3/yli/data/WORD-V0.1.0/"
    dataset_name = "WORD"
    json_name = os.path.join(output_json_dir, f"{dataset_name}_{n_folds}_folds.json")
    masks = search_image_files(os.path.join(base_url, "labelsTr"), ".nii.gz")
    masks += search_image_files(os.path.join(base_url, "labelsTs"), ".nii.gz")
    masks += search_image_files(os.path.join(base_url, "labelsVal"), ".nii.gz")
    images, labels = [], []
    for mask in masks:
        rel_mask = os.path.relpath(mask, base_url)
        labels.append(rel_mask)
        idx = re.compile(r"word_(\d+).nii.gz").search(rel_mask)[1]
        img_name = f"word_{idx}.nii.gz"
        for f in ["imagesTr", "imagesTs", "imagesVal"]:
            if os.path.exists(os.path.join(base_url, f, img_name)):
                images.append(os.path.join(f, img_name))
                break
        filtering_files(base_url, images, labels)
    original_label_dict = {
        1: "liver",
        2: "spleen",
        3: "left_kidney",
        4: "right_kidney",
        5: "stomach",
        6: "gallbladder",
        7: "esophagus",
        8: "pancreas",
        9: "duodenum",
        10: "colon",
        11: "intestine",
        12: "adrenal",
        13: "rectum",
        14: "bladder",
        15: "Head_of_femur_L",
        16: "Head_of_femur_R",
    }
    label_dict = {
        1: "liver",
        2: "spleen",
        3: "left kidney",
        4: "right kidney",
        5: "stomach",
        6: "gallbladder",
        7: "esophagus",
        8: "pancreas",
        9: "duodenum",
        10: "colon",
        11: "intestine",
        12: "adrenal gland",
        13: "rectum",
        14: "bladder",
        15: "left head of femur",
        16: "right head of femur",
    }
    create_splits_and_write_json(
        images,
        labels,
        test_ratio,
        n_folds,
        json_name,
        seed,
        label_dict,
        original_label_dict,
    )


if __name__ == "__main__":
    pprint(_make_funcs)
    for func_name, f in _make_funcs.items():
        print(f"running {func_name}")
        f()

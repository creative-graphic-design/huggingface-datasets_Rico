import json
import math
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import datasets as ds
from PIL import Image

_DESCRIPTION = ""

_CITATION = ""

_HOMEPAGE = ""

_LICENSE = ""

_URLS = {
    "tasks": {
        "ui-screenshots-and-view-hierarchies": "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz",
        "ui-layout-vectors": "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/ui_layout_vectors.zip",
        "interaction-traces": "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/traces.tar.gz",
        "animations": "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/animations.tar.gz",
        "ui-screenshots-and-hierarchies-with-semantic-annotations": "https://storage.cloud.google.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/semantic_annotations.zip",
    },
    "metadata": {
        "ui-metadata": "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/ui_details.csv",
        "play-store-bmetadata": "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/app_details.csv",
    },
}


def flatten_children(
    children,
    children_id: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
):
    result = result or defaultdict(list)
    if children is None:
        return result

    children_id = children_id or 0

    for child in children:
        if not child:
            continue

        if "children" not in child:
            continue

        result = flatten_children(
            children=child.pop("children"),
            children_id=children_id + 1,
            result=result,
        )

        if "resource-id" not in child:
            child["resource-id"] = None
        if "package" not in child:
            child["package"] = None
        if "rel-bounds" not in child:
            child["rel-bounds"] = None

        assert result is not None
        result[f"children_{children_id}"].append(child)

    return result


@dataclass
class RicoConfig(ds.BuilderConfig):
    train_ratio: float = 0.85
    validation_ratio: float = 0.05
    test_ratio: float = 0.10
    random_state: int = 0

    def __post_init__(self):
        assert self.train_ratio + self.validation_ratio + self.test_ratio == 1.0


class RicoDataset(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIG_CLASS = RicoConfig
    BUILDER_CONFIGS = [
        RicoConfig(
            name="ui-screenshots-and-view-hierarchies",
            version=VERSION,
            description="Contains 66k+ unique UI screens",
        ),
        RicoConfig(
            name="ui-layout-vectors",
            version=VERSION,
            description="Contains 64-dimensional vector representations for each UI screen that encode layout based on the distribution of text and images.",
        ),
        RicoConfig(
            name="interraction-traces",
            version=VERSION,
            description="Contains user interaction traces organized by app.",
        ),
        RicoConfig(
            name="animations",
            version=VERSION,
            description="Contains GIFs that demonstrate how screens animated in response to a user interaction; follows the same folder structure introduced for interaction traces.",
        ),
        RicoConfig(
            name="ui-screenshots-and-hierarchies-with-semantic-annotations",
            version=VERSION,
            description="Contains 66k+ UI screens and hierarchies augmented with semantic annotations that describe what elements on the screen mean and how they are used.",
        ),
    ]

    def _info(self) -> ds.DatasetInfo:
        activity_class = {
            "scrollable-horizontal": ds.Value("bool"),
            "draw": ds.Value("bool"),
            "ancestors": ds.Sequence(ds.Value("string")),
            "clickable": ds.Value("bool"),
            "pressed": ds.Value("string"),
            "focusable": ds.Value("bool"),
            "long-clickable": ds.Value("bool"),
            "enabled": ds.Value("bool"),
            "bounds": ds.Sequence(ds.Value("int64")),
            "visibility": ds.Value("string"),
            "content-desc": ds.Sequence(ds.Value("string")),
            "rel-bounds": ds.Sequence(ds.Value("int64")),
            "focused": ds.Value("bool"),
            "selected": ds.Value("bool"),
            "scrollable-vertical": ds.Value("bool"),
            "adapter-view": ds.Value("bool"),
            "abs-pos": ds.Value("bool"),
            "pointer": ds.Value("string"),
            "class": ds.Value("string"),
            "visible-to-user": ds.Value("bool"),
            "resource-id": ds.Value("string"),
            "package": ds.Value("string"),
        }
        features = ds.Features(
            {
                "activity_name": ds.Value("string"),
                "screenshot": ds.Image(),
                "activity": {
                    "root": activity_class,
                    "children": ds.Sequence(ds.Sequence(activity_class)),
                    "added_fragments": ds.Sequence(ds.Value("string")),
                    "active_fragments": ds.Sequence(ds.Value("string")),
                },
                "is_keyboard_deployed": ds.Value("bool"),
                "request_id": ds.Value("string"),
            }
        )
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            features=features,
        )

    def _split_generators(self, dl_manager: ds.DownloadManager):
        task_base_dir = dl_manager.download_and_extract(
            _URLS["tasks"][self.config.name]
        )
        metadata_files = dl_manager.download_and_extract(_URLS["metadata"])

        task_dir = pathlib.Path(task_base_dir) / "combined"
        json_files = [f for f in task_dir.iterdir() if f.suffix == ".json"]

        num_samples = len(json_files)
        num_tng = math.ceil(num_samples * self.config.train_ratio)  # type: ignore
        num_val = math.ceil(num_samples * self.config.validation_ratio)  # type: ignore
        num_tst = math.ceil(num_samples * self.config.test_ratio)  # type: ignore

        tng_files = json_files[:num_tng]
        val_files = json_files[num_tng : num_tng + num_val]
        tst_files = json_files[num_tng + num_val : num_tng + num_val + num_tst]

        assert len(tng_files) + len(val_files) + len(tst_files) == num_samples

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,  # type: ignore
                gen_kwargs={"json_files": tng_files},
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,  # type: ignore
                gen_kwargs={"json_files": val_files},
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,  # type: ignore
                gen_kwargs={"json_files": tst_files},
            ),
        ]

    def _generate_examples(self, json_files: List[pathlib.Path]):
        for i, json_file in enumerate(json_files):
            with json_file.open("r") as rf:
                json_dict = json.load(rf)

                if "resource-id" not in json_dict["activity"]["root"]:
                    json_dict["activity"]["root"]["resource-id"] = None
                if "package" not in json_dict["activity"]["root"]:
                    json_dict["activity"]["root"]["package"] = None
                if "rel-bounds" not in json_dict["activity"]["root"]:
                    json_dict["activity"]["root"]["rel-bounds"] = None

                children = flatten_children(
                    children=json_dict["activity"]["root"].pop("children")
                )
                json_dict["activity"]["children"] = [v for v in children.values()]

                json_dict["screenshot"] = Image.open(
                    json_file.parent / f"{json_file.stem}.jpg"
                )

                yield i, json_dict

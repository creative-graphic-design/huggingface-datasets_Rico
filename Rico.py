import abc
import json
import math
import pathlib
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import datasets as ds
import numpy as np
import pandas as pd
from datasets.utils.logging import get_logger
from PIL import Image
from PIL.Image import Image as PilImage

logger = get_logger(__name__)

JsonDict = Dict[str, Any]

_DESCRIPTION = """
THE DATASET: We mined over 9.3k free Android apps from 27 categories to create the Rico dataset. Apps in the dataset had an average user rating of 4.1. The Rico dataset contains visual, textual, structural, and interactive design properties of more than 66k unique UI screens and 3M UI elements.
"""

_CITATION = """\
@inproceedings{deka2017rico,
  title={Rico: A mobile app dataset for building data-driven design applications},
  author={Deka, Biplab and Huang, Zifeng and Franzen, Chad and Hibschman, Joshua and Afergan, Daniel and Li, Yang and Nichols, Jeffrey and Kumar, Ranjitha},
  booktitle={Proceedings of the 30th annual ACM symposium on user interface software and technology},
  pages={845--854},
  year={2017}
}
"""

_HOMEPAGE = "http://www.interactionmining.org/rico.html"

_LICENSE = "Unknown"


def to_snake_case(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub("__([A-Z])", r"_\1", name)
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()


class TrainValidationTestSplit(TypedDict):
    train: List[Any]
    validation: List[Any]
    test: List[Any]


class UiLayoutVectorSample(TypedDict):
    vector: np.ndarray
    name: str


@dataclass(eq=True)
class RicoProcessor(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_features(self) -> ds.Features:
        raise NotImplementedError

    @abc.abstractmethod
    def load_examples(self, *args, **kwargs) -> List[Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def split_generators(self, *args, **kwargs) -> List[ds.SplitGenerator]:
        raise NotImplementedError

    @abc.abstractmethod
    def generate_examples(self, examples: List[Any]):
        raise NotImplementedError


class RicoTaskProcessor(RicoProcessor, metaclass=abc.ABCMeta):
    def _flatten_children(
        self,
        children,
        children_id: Optional[int] = None,
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

            result = self._flatten_children(
                children=child.pop("children"),
                children_id=children_id + 1,
                result=result,
            )
            assert result is not None
            result[f"children_{children_id}"].append(child)

        return result

    def _load_image(self, file_path: pathlib.Path) -> PilImage:
        logger.debug(f"Load from {file_path}")
        return Image.open(file_path)

    def _load_json(self, file_path: pathlib.Path) -> JsonDict:
        logger.debug(f"Load from {file_path}")
        with file_path.open("r") as rf:
            json_dict = json.load(rf)
        return json_dict

    def _split_dataset(
        self,
        examples: List[Any],
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
    ) -> TrainValidationTestSplit:
        assert train_ratio + validation_ratio + test_ratio == 1.0
        num_examples = len(examples)

        num_tng = math.ceil(num_examples * train_ratio)  # type: ignore
        num_val = math.ceil(num_examples * validation_ratio)  # type: ignore
        num_tst = math.ceil(num_examples * test_ratio)  # type: ignore

        tng_examples = examples[:num_tng]
        val_examples = examples[num_tng : num_tng + num_val]
        tst_examples = examples[num_tng + num_val : num_tng + num_val + num_tst]
        assert len(tng_examples) + len(val_examples) + len(tst_examples) == num_examples

        return {
            "train": tng_examples,
            "validation": val_examples,
            "test": tst_examples,
        }

    def _load_and_split_dataset(
        self,
        base_dir: pathlib.Path,
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
    ) -> TrainValidationTestSplit:
        examples = self.load_examples(base_dir)
        return self._split_dataset(
            examples=examples,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )

    def split_generators(
        self,
        base_dir: pathlib.Path,
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
    ) -> List[ds.SplitGenerator]:
        split_examples = self._load_and_split_dataset(
            base_dir=pathlib.Path(base_dir),
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,  # type: ignore
                gen_kwargs={"examples": split_examples["train"]},
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,  # type: ignore
                gen_kwargs={"examples": split_examples["validation"]},
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,  # type: ignore
                gen_kwargs={"examples": split_examples["test"]},
            ),
        ]

    @abc.abstractmethod
    def load_examples(self, base_dir: pathlib.Path) -> List[Any]:
        raise NotImplementedError


class RicoMetadataProcessor(RicoProcessor, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def load_examples(self, csv_file: pathlib.Path) -> List[Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def split_generators(self, csv_file: pathlib.Path) -> List[ds.SplitGenerator]:
        raise NotImplementedError


@dataclass
class ActivityClass(object):
    abs_pos: bool
    adapter_view: bool
    ancestors: List[str]
    bounds: Tuple[int, int, int, int]
    clickable: bool
    content_desc: List[str]
    draw: bool
    enabled: bool
    focused: bool
    focusable: bool
    klass: str
    long_clickable: bool
    pressed: bool
    pointer: str
    scrollable_horizontal: bool
    scrollable_vertical: bool
    selected: bool
    visibility: str
    visible_to_user: bool

    package: Optional[str] = None
    resource_id: Optional[str] = None
    rel_bounds: Optional[Tuple[int, int, int, int]] = None

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "ActivityClass":
        json_dict = {k.replace("-", "_"): v for k, v in json_dict.items()}
        json_dict["klass"] = json_dict.pop("class")
        return cls(**json_dict)


@dataclass
class UiComponent(object):
    ancestors: List[str]
    bounds: Tuple[int, int, int, int]
    component_label: str
    clickable: bool
    klass: str

    icon_class: Optional[str] = None
    resource_id: Optional[str] = None

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "UiComponent":
        json_dict = {
            to_snake_case(k.replace("-", "_")): v for k, v in json_dict.items()
        }
        json_dict["klass"] = json_dict.pop("class")
        return cls(**json_dict)


@dataclass
class Activity(object):
    root: ActivityClass
    children: List[List[ActivityClass]]
    added_fragments: List[str]
    active_fragments: List[str]

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "Activity":
        root = ActivityClass.from_dict(json_dict.pop("root"))
        children = [
            [
                ActivityClass.from_dict(activity_class)
                for activity_class in activity_classes
            ]
            for activity_classes in json_dict.pop("children")
        ]
        return cls(root=root, children=children, **json_dict)


@dataclass
class InteractionTracesData(object):
    activity_name: str
    activity: Activity
    is_keyboard_deployed: str
    request_id: str

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "InteractionTracesData":
        activity_dict = json_dict.pop("activity")
        activity = Activity.from_dict(activity_dict)
        return cls(activity=activity, **json_dict)


@dataclass
class UiScreenshotsAndViewHierarchiesData(InteractionTracesData):
    screenshot: PilImage

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "UiScreenshotsAndViewHierarchiesData":
        activity_dict = json_dict.pop("activity")
        activity = Activity.from_dict(activity_dict)
        return cls(activity=activity, **json_dict)


@dataclass
class UiScreenshotsAndHierarchiesWithSemanticAnnotationsData(object):
    ancestors: List[str]
    klass: str
    bounds: Tuple[int, int, int, int]
    clickable: bool
    children: List[List[UiComponent]]
    screenshot: PilImage

    @classmethod
    def from_dict(
        cls, json_dict: JsonDict
    ) -> "UiScreenshotsAndHierarchiesWithSemanticAnnotationsData":
        json_dict["klass"] = json_dict.pop("class")
        children = [
            [UiComponent.from_dict(ui_component) for ui_component in ui_components]
            for ui_components in json_dict.pop("children")
        ]
        return cls(children=children, **json_dict)


@dataclass
class Gesture(object):
    ui_id: int
    xy: List[Tuple[float, float]]

    @classmethod
    def from_dict_to_gestures(cls, json_dict: JsonDict) -> List["Gesture"]:
        return [Gesture(ui_id=int(k), xy=v) for k, v in json_dict.items()]


class InteractionTracesProcessor(RicoTaskProcessor):
    def get_activity_class_features_dict(self):
        return {
            "abs_pos": ds.Value("bool"),
            "adapter_view": ds.Value("bool"),
            "ancestors": ds.Sequence(ds.Value("string")),
            "bounds": ds.Sequence(ds.Value("int64")),
            "clickable": ds.Value("bool"),
            "content_desc": ds.Sequence(ds.Value("string")),
            "draw": ds.Value("bool"),
            "enabled": ds.Value("bool"),
            "focusable": ds.Value("bool"),
            "focused": ds.Value("bool"),
            "klass": ds.Value("string"),
            "long_clickable": ds.Value("bool"),
            "package": ds.Value("string"),
            "pressed": ds.Value("string"),
            "pointer": ds.Value("string"),
            "rel_bounds": ds.Sequence(ds.Value("int64")),
            "resource_id": ds.Value("string"),
            "scrollable_horizontal": ds.Value("bool"),
            "scrollable_vertical": ds.Value("bool"),
            "selected": ds.Value("bool"),
            "visibility": ds.Value("string"),
            "visible_to_user": ds.Value("bool"),
        }

    def get_activity_features_dict(self, activity_class):
        return {
            "activity_name": ds.Value("string"),
            "activity": {
                "root": activity_class,
                "children": ds.Sequence(ds.Sequence(activity_class)),
                "added_fragments": ds.Sequence(ds.Value("string")),
                "active_fragments": ds.Sequence(ds.Value("string")),
            },
            "is_keyboard_deployed": ds.Value("bool"),
            "request_id": ds.Value("string"),
        }

    def get_features(self) -> ds.Features:
        activity_class = self.get_activity_class_features_dict()
        activity = self.get_activity_features_dict(activity_class)
        return ds.Features(
            {
                "screenshots": ds.Sequence(ds.Image()),
                "view_hierarchies": ds.Sequence(activity),
                "gestures": ds.Sequence(
                    {
                        "ui_id": ds.Value("int32"),
                        "xy": ds.Sequence(ds.Sequence(ds.Value("float32"))),
                    }
                ),
            }
        )

    def load_examples(self, base_dir: pathlib.Path) -> List[pathlib.Path]:
        task_dir = base_dir / "filtered_traces"
        return [d for d in task_dir.iterdir() if d.is_dir()]

    def generate_examples(self, examples: List[pathlib.Path]):
        idx = 0
        for trace_base_dir in examples:
            for trace_dir in trace_base_dir.iterdir():
                screenshots_dir = trace_dir / "screenshots"
                screenshots = [
                    self._load_image(f)
                    for f in screenshots_dir.iterdir()
                    if not f.name.startswith("._")
                ]

                view_hierarchies_dir = trace_dir / "view_hierarchies"
                view_hierarchies_json_files = [
                    f
                    for f in view_hierarchies_dir.iterdir()
                    if f.suffix == ".json" and not f.name.startswith("._")
                ]
                view_hierarchies_jsons = []
                for json_file in view_hierarchies_json_files:
                    json_dict = self._load_json(json_file)
                    if json_dict is None:
                        logger.warning(f"Invalid json file: {json_file}")
                        continue

                    children = self._flatten_children(
                        children=json_dict["activity"]["root"].pop("children")
                    )

                    json_dict["activity"]["children"] = [v for v in children.values()]
                    data = InteractionTracesData.from_dict(json_dict)
                    view_hierarchies_jsons.append(asdict(data))

                gestures_json = trace_dir / "gestures.json"
                with gestures_json.open("r") as rf:
                    gestures_dict = json.load(rf)
                gestures = Gesture.from_dict_to_gestures(gestures_dict)

                example = {
                    "screenshots": screenshots,
                    "view_hierarchies": view_hierarchies_jsons,
                    "gestures": [asdict(gesture) for gesture in gestures],
                }
                yield idx, example
                idx += 1


class UiScreenshotsAndViewHierarchiesProcessor(InteractionTracesProcessor):
    def get_features(self) -> ds.Features:
        activity_class = self.get_activity_class_features_dict()
        activity = {
            "screenshot": ds.Image(),
            **self.get_activity_features_dict(activity_class),
        }
        return ds.Features(activity)

    def load_examples(self, base_dir: pathlib.Path) -> List[Any]:
        task_dir = base_dir / "combined"
        json_files = [f for f in task_dir.iterdir() if f.suffix == ".json"]
        return json_files

    def generate_examples(self, examples: List[pathlib.Path]):
        for i, json_file in enumerate(examples):
            with json_file.open("r") as rf:
                json_dict = json.load(rf)
                children = self._flatten_children(
                    children=json_dict["activity"]["root"].pop("children")
                )
                json_dict["activity"]["children"] = [v for v in children.values()]
                json_dict["screenshot"] = self._load_image(
                    json_file.parent / f"{json_file.stem}.jpg"
                )
                data = UiScreenshotsAndViewHierarchiesData.from_dict(json_dict)
                example = asdict(data)
                yield i, example


class UiLayoutVectorsProcessor(RicoTaskProcessor):
    def get_features(self) -> ds.Features:
        return ds.Features(
            {"vector": ds.Sequence(ds.Value("float32")), "name": ds.Value("string")}
        )

    def _load_ui_vectors(self, file_path: pathlib.Path) -> np.ndarray:
        logger.debug(f"Load from {file_path}")
        ui_vectors = np.load(file_path)
        assert ui_vectors.shape[1] == 64
        return ui_vectors

    def _load_ui_names(self, file_path: pathlib.Path) -> List[str]:
        with file_path.open("r") as rf:
            json_dict = json.load(rf)
        return json_dict["ui_names"]

    def load_examples(self, base_dir: pathlib.Path) -> List[UiLayoutVectorSample]:
        task_dir = base_dir / "ui_layout_vectors"
        ui_vectors = self._load_ui_vectors(file_path=task_dir / "ui_vectors.npy")
        ui_names = self._load_ui_names(file_path=task_dir / "ui_names.json")
        assert len(ui_vectors) == len(ui_names)

        return [
            {"vector": vector, "name": name}
            for vector, name in zip(ui_vectors, ui_names)
        ]

    def generate_examples(self, examples: List[UiLayoutVectorSample]):
        for i, sample in enumerate(examples):
            sample["vector"] = sample["vector"].tolist()
            yield i, sample


class AnimationsProcessor(RicoTaskProcessor):
    def get_features(self) -> ds.Features:
        raise NotImplementedError

    def load_examples(self, base_dir: pathlib.Path) -> List[Any]:
        raise NotImplementedError

    def generate_examples(self, examples: List[Any]):
        raise NotImplementedError


class UiScreenshotsAndHierarchiesWithSemanticAnnotationsProcessor(RicoTaskProcessor):
    def get_features(self) -> ds.Features:
        ui_component = {
            "ancestors": ds.Sequence(ds.Value("string")),
            "bounds": ds.Sequence(ds.Value("int64")),
            "component_label": ds.ClassLabel(
                num_classes=25,
                names=[
                    "Text",
                    "Image",
                    "Icon",
                    "Text Button",
                    "List Item",
                    "Input",
                    "Background Image",
                    "Card",
                    "Web View",
                    "Radio Button",
                    "Drawer",
                    "Checkbox",
                    "Advertisement",
                    "Modal",
                    "Pager Indicator",
                    "Slider",
                    "On/Off Switch",
                    "Button Bar",
                    "Toolbar",
                    "Number Stepper",
                    "Multi-Tab",
                    "Date Picker",
                    "Map View",
                    "Video",
                    "Bottom Navigation",
                ],
            ),
            "clickable": ds.Value("bool"),
            "klass": ds.Value("string"),
            "icon_class": ds.Value("string"),
            "resource_id": ds.Value("string"),
        }
        return ds.Features(
            {
                "ancestors": ds.Sequence(ds.Value("string")),
                "klass": ds.Value("string"),
                "bounds": ds.Sequence(ds.Value("int64")),
                "clickable": ds.Value("bool"),
                "children": ds.Sequence(ds.Sequence(ui_component)),
                "screenshot": ds.Image(),
            }
        )

    def load_examples(self, base_dir: pathlib.Path) -> List[Any]:
        task_dir = base_dir / "semantic_annotations"
        json_files = [f for f in task_dir.iterdir() if f.suffix == ".json"]
        return json_files

    def generate_examples(self, examples: List[pathlib.Path]):
        for i, json_file in enumerate(examples):
            with json_file.open("r") as rf:
                json_dict = json.load(rf)

                children = self._flatten_children(children=json_dict.pop("children"))
                json_dict["children"] = [v for v in children.values()]
                json_dict["screenshot"] = self._load_image(
                    json_file.parent / f"{json_file.stem}.png"
                )
                data = UiScreenshotsAndHierarchiesWithSemanticAnnotationsData.from_dict(
                    json_dict
                )
                yield i, asdict(data)


class UiMetadataProcessor(RicoMetadataProcessor):
    def get_features(self) -> ds.Features:
        return ds.Features(
            {
                "ui_number": ds.Value("int32"),
                "app_package_name": ds.Value("string"),
                "interaction_trace_number": ds.Value("string"),
                "ui_number_in_trace": ds.Value("string"),
            }
        )

    def load_examples(self, csv_file: pathlib.Path) -> List[Any]:
        df = pd.read_csv(csv_file)  # 66261 col
        df.columns = ["_".join(col.split()) for col in df.columns.str.lower()]
        return df.to_dict(orient="records")

    def split_generators(
        self, csv_file: pathlib.Path, **kwargs
    ) -> List[ds.SplitGenerator]:
        metadata = self.load_examples(csv_file)
        return [ds.SplitGenerator(name="metadata", gen_kwargs={"examples": metadata})]

    def generate_examples(self, examples: List[Any]):
        for i, metadata in enumerate(examples):
            yield i, metadata


class PlayStoreMetadataProcessor(RicoMetadataProcessor):
    def get_features(self) -> ds.Features:
        return ds.Features(
            {
                "app_package_name": ds.Value("string"),
                "play_store_name": ds.Value("string"),
                "category": ds.ClassLabel(
                    num_classes=27,
                    names=[
                        "Books & Reference",
                        "Comics",
                        "Health & Fitness",
                        "Social",
                        "Entertainment",
                        "Weather",
                        "Communication",
                        "Sports",
                        "News & Magazines",
                        "Finance",
                        "Shopping",
                        "Education",
                        "Travel & Local",
                        "Business",
                        "Medical",
                        "Beauty",
                        "Food & Drink",
                        "Dating",
                        "Auto & Vehicles",
                        "Music & Audio",
                        "House & Home",
                        "Maps & Navigation",
                        "Lifestyle",
                        "Art & Design",
                        "Parenting",
                        "Events",
                        "Video Players & Editors",
                    ],
                ),
                "average_rating": ds.Value("float32"),
                "number_of_ratings": ds.Value("int32"),
                "number_of_downloads": ds.ClassLabel(
                    num_classes=15,
                    names=[
                        "100,000 - 500,000",
                        "10,000 - 50,000",
                        "50,000,000 - 100,000,000",
                        "50,000 - 100,000",
                        "1,000,000 - 5,000,000",
                        "5,000,000 - 10,000,000",
                        "500,000 - 1,000,000",
                        "1,000 - 5,000",
                        "10,000,000 - 50,000,000",
                        "5,000 - 10,000",
                        "100,000,000 - 500,000,000",
                        "500,000,000 - 1,000,000,000",
                        "500 - 1,000",
                        "1,000,000,000 - 5,000,000,000",
                        "100 - 500",
                    ],
                ),
                "date_updated": ds.Value("string"),
                "icon_url": ds.Value("string"),
            }
        )

    def cleanup_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(
            number_of_downloads=df["number_of_downloads"].str.strip(),
            number_of_ratings=df["number_of_ratings"]
            .str.replace('"', "")
            .str.strip()
            .astype(int),
        )

        def remove_noisy_data(df: pd.DataFrame) -> pd.DataFrame:
            old_num = len(df)
            df = df[
                (df["category"] != "000 - 1")
                | (df["number_of_downloads"] != "January 10, 2015")
            ]
            new_num = len(df)
            assert new_num == old_num - 1
            return df

        df = remove_noisy_data(df)

        return df

    def load_examples(self, csv_file: pathlib.Path) -> List[Any]:
        df = pd.read_csv(csv_file)
        df.columns = ["_".join(col.split()) for col in df.columns.str.lower()]
        df = self.cleanup_metadata(df)
        return df.to_dict(orient="records")

    def split_generators(
        self, csv_file: pathlib.Path, **kwargs
    ) -> List[ds.SplitGenerator]:
        metadata = self.load_examples(csv_file)
        return [ds.SplitGenerator(name="metadata", gen_kwargs={"examples": metadata})]

    def generate_examples(self, examples: List[Any]):
        for i, metadata in enumerate(examples):
            yield i, metadata


@dataclass
class RicoConfig(ds.BuilderConfig):
    train_ratio: float = 0.85
    validation_ratio: float = 0.05
    test_ratio: float = 0.10
    random_state: int = 0
    data_url: Optional[str] = None
    processor: Optional[RicoProcessor] = None

    def __post_init__(self):
        assert self.data_url is not None
        assert self.processor is not None
        assert self.train_ratio + self.validation_ratio + self.test_ratio == 1.0


class RicoDataset(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIGS = [
        RicoConfig(
            name="ui-screenshots-and-view-hierarchies",
            version=VERSION,
            description="Contains 66k+ unique UI screens",
            data_url="https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz",
            processor=UiScreenshotsAndViewHierarchiesProcessor(),
        ),
        RicoConfig(
            name="ui-layout-vectors",
            version=VERSION,
            description="Contains 64-dimensional vector representations for each UI screen that encode layout based on the distribution of text and images.",
            data_url="https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/ui_layout_vectors.zip",
            processor=UiLayoutVectorsProcessor(),
        ),
        RicoConfig(
            name="interaction-traces",
            version=VERSION,
            description="Contains user interaction traces organized by app.",
            data_url="https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/traces.tar.gz",
            processor=InteractionTracesProcessor(),
        ),
        RicoConfig(
            name="animations",
            version=VERSION,
            description="Contains GIFs that demonstrate how screens animated in response to a user interaction; follows the same folder structure introduced for interaction traces.",
            data_url="https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/animations.tar.gz",
            processor=AnimationsProcessor(),
        ),
        RicoConfig(
            name="ui-screenshots-and-hierarchies-with-semantic-annotations",
            version=VERSION,
            description="Contains 66k+ UI screens and hierarchies augmented with semantic annotations that describe what elements on the screen mean and how they are used.",
            data_url="https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/semantic_annotations.zip",
            processor=UiScreenshotsAndHierarchiesWithSemanticAnnotationsProcessor(),
        ),
        RicoConfig(
            name="ui-metadata",
            version=VERSION,
            description="Contains metadata about each UI screen: the name of the app it came from, the user interaction trace within that app.",
            data_url="https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/ui_details.csv",
            processor=UiMetadataProcessor(),
        ),
        RicoConfig(
            name="play-store-metadata",
            version=VERSION,
            description="Contains metadata about the apps in the dataset including an app’s category, average rating, number of ratings, and number of downloads.",
            data_url="https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/app_details.csv",
            processor=PlayStoreMetadataProcessor(),
        ),
    ]

    def _info(self) -> ds.DatasetInfo:
        processor: RicoProcessor = self.config.processor
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            features=processor.get_features(),
        )

    def _split_generators(self, dl_manager: ds.DownloadManager):
        config: RicoConfig = self.config
        assert config.processor is not None
        processor: RicoProcessor = config.processor

        return processor.split_generators(
            dl_manager.download_and_extract(self.config.data_url),
            train_ratio=config.train_ratio,
            validation_ratio=config.validation_ratio,
            test_ratio=config.test_ratio,
        )

    def _generate_examples(self, **kwargs):
        config: RicoConfig = self.config
        assert config.processor is not None
        processor: RicoProcessor = config.processor
        yield from processor.generate_examples(**kwargs)

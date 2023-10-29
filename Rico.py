import abc
import json
import math
import pathlib
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import datasets as ds
from PIL import Image
from PIL.Image import Image as PilImage

JsonDict = Dict[str, Any]

_DESCRIPTION = ""

_CITATION = ""

_HOMEPAGE = ""

_LICENSE = ""

_METADATA_URLS = {
    "metadata": {
        "ui-metadata": "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/ui_details.csv",
        "play-store-bmetadata": "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/app_details.csv",
    },
}


def to_snake_case(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub("__([A-Z])", r"_\1", name)
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()


class TrainValidationTestSplit(TypedDict):
    train: List[Any]
    validation: List[Any]
    test: List[Any]


@dataclass(eq=True)
class RicoTaskProcessor(object, metaclass=abc.ABCMeta):
    def flatten_children(
        self,
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

            result = self.flatten_children(
                children=child.pop("children"),
                children_id=children_id + 1,
                result=result,
            )
            assert result is not None
            result[f"children_{children_id}"].append(child)

        return result

    def load_image(self, file_path: pathlib.Path) -> PilImage:
        return Image.open(file_path)

    def split_dataset(
        self,
        samples: List[Any],
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
    ) -> TrainValidationTestSplit:
        assert train_ratio + validation_ratio + test_ratio == 1.0
        num_samples = len(samples)

        num_tng = math.ceil(num_samples * train_ratio)  # type: ignore
        num_val = math.ceil(num_samples * validation_ratio)  # type: ignore
        num_tst = math.ceil(num_samples * test_ratio)  # type: ignore

        tng_samples = samples[:num_tng]
        val_samples = samples[num_tng : num_tng + num_val]
        tst_samples = samples[num_tng + num_val : num_tng + num_val + num_tst]
        assert len(tng_samples) + len(val_samples) + len(tst_samples) == num_samples

        return {
            "train": tng_samples,
            "validation": val_samples,
            "test": tst_samples,
        }

    def load_and_split_dataset(
        self,
        base_dir: pathlib.Path,
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
    ) -> TrainValidationTestSplit:
        samples = self.load_samples(base_dir=base_dir)
        return self.split_dataset(
            samples=samples,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )

    @abc.abstractmethod
    def load_samples(self, base_dir: pathlib.Path) -> List[Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_features(self) -> ds.Features:
        raise NotImplementedError

    @abc.abstractmethod
    def generate_examples(self, samples: List[Any]):
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
class UiScreenshotsAndViewHierarchiesData(object):
    activity_name: str
    activity: Activity
    is_keyboard_deployed: str
    request_id: str
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


class UiScreenshotsAndViewHierarchiesProcessor(RicoTaskProcessor):
    def get_features(self) -> ds.Features:
        activity_class = {
            "abs_pos": ds.Value("bool"),
            "adapter_view": ds.Value("bool"),
            "ancestors": ds.Sequence(ds.Value("string")),
            "bounds": ds.Sequence(ds.Value("int64")),
            "clickable": ds.Value("bool"),
            "content_desc": ds.Sequence(ds.Value("string")),
            "draw": ds.Value("bool"),
            "enabled": ds.Value("bool"),
            "focused": ds.Value("bool"),
            "focusable": ds.Value("bool"),
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
        return features

    def load_samples(self, base_dir: pathlib.Path) -> List[Any]:
        task_dir = base_dir / "combined"
        json_files = [f for f in task_dir.iterdir() if f.suffix == ".json"]
        return json_files

    def generate_examples(self, samples: List[pathlib.Path]):
        for i, json_file in enumerate(samples):
            with json_file.open("r") as rf:
                json_dict = json.load(rf)
                children = self.flatten_children(
                    children=json_dict["activity"]["root"].pop("children")
                )
                json_dict["activity"]["children"] = [v for v in children.values()]
                json_dict["screenshot"] = self.load_image(
                    json_file.parent / f"{json_file.stem}.jpg"
                )
                data = UiScreenshotsAndViewHierarchiesData.from_dict(json_dict)

                yield i, asdict(data)


class UiLayoutVectorsProcessor(RicoTaskProcessor):
    def get_features(self) -> ds.Features:
        return ds.Features()

    def load_samples(self, base_dir: pathlib.Path) -> List[Any]:
        raise NotImplementedError

    def generate_examples(self, samples: List[Any]):
        raise NotImplementedError


class InteractionTracesProcessor(RicoTaskProcessor):
    def get_features(self) -> ds.Features:
        raise NotImplementedError

    def load_samples(self, base_dir: pathlib.Path) -> List[Any]:
        raise NotImplementedError

    def generate_examples(self, samples: List[Any]):
        raise NotImplementedError


class AnimationsProcessor(RicoTaskProcessor):
    def get_features(self) -> ds.Features:
        raise NotImplementedError

    def load_samples(self, base_dir: pathlib.Path) -> List[Any]:
        raise NotImplementedError

    def generate_examples(self, samples: List[Any]):
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

    def load_samples(self, base_dir: pathlib.Path) -> List[Any]:
        task_dir = base_dir / "semantic_annotations"
        json_files = [f for f in task_dir.iterdir() if f.suffix == ".json"]
        return json_files

    def generate_examples(self, samples: List[pathlib.Path]):
        for i, json_file in enumerate(samples):
            with json_file.open("r") as rf:
                json_dict = json.load(rf)

                children = self.flatten_children(children=json_dict.pop("children"))
                json_dict["children"] = [v for v in children.values()]
                json_dict["screenshot"] = self.load_image(
                    json_file.parent / f"{json_file.stem}.png"
                )
                data = UiScreenshotsAndHierarchiesWithSemanticAnnotationsData.from_dict(
                    json_dict
                )
                yield i, asdict(data)


@dataclass
class RicoConfig(ds.BuilderConfig):
    train_ratio: float = 0.85
    validation_ratio: float = 0.05
    test_ratio: float = 0.10
    random_state: int = 0
    data_url: Optional[str] = None
    processor: Optional[RicoTaskProcessor] = None

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
    ]

    def _info(self) -> ds.DatasetInfo:
        processor: RicoTaskProcessor = self.config.processor
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            features=processor.get_features(),
        )

    def _split_generators(self, dl_manager: ds.DownloadManager):
        task_base_dir = dl_manager.download_and_extract(self.config.data_url)

        metadata_files = dl_manager.download_and_extract(_METADATA_URLS["metadata"])

        processor: RicoTaskProcessor = self.config.processor
        split_samples = processor.load_and_split_dataset(
            base_dir=pathlib.Path(task_base_dir),
            train_ratio=self.config.train_ratio,
            validation_ratio=self.config.validation_ratio,
            test_ratio=self.config.test_ratio,
        )

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,  # type: ignore
                gen_kwargs={"samples": split_samples["train"]},
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,  # type: ignore
                gen_kwargs={"samples": split_samples["validation"]},
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,  # type: ignore
                gen_kwargs={"samples": split_samples["test"]},
            ),
        ]

    def _generate_examples(self, samples: List[Any]):
        processor: RicoTaskProcessor = self.config.processor
        yield from processor.generate_examples(samples)

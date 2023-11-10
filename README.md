---
annotations_creators:
- found
language:
- en
language_creators:
- found
license:
- unknown
multilinguality:
- monolingual
pretty_name: Rico
size_categories: []
source_datasets:
- original
tags:
- graphic design
task_categories:
- other
task_ids: []
---

# Dataset Card for Rico

[![CI](https://github.com/shunk031/huggingface-datasets_Rico/actions/workflows/ci.yaml/badge.svg)](https://github.com/shunk031/huggingface-datasets_Rico/actions/workflows/ci.yaml)

## Table of Contents
- [Dataset Card Creation Guide](#dataset-card-creation-guide)
  - [Table of Contents](#table-of-contents)
  - [Dataset Description](#dataset-description)
    - [Dataset Summary](#dataset-summary)
    - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
    - [Languages](#languages)
  - [Dataset Structure](#dataset-structure)
    - [Data Instances](#data-instances)
    - [Data Fields](#data-fields)
    - [Data Splits](#data-splits)
  - [Dataset Creation](#dataset-creation)
    - [Curation Rationale](#curation-rationale)
    - [Source Data](#source-data)
      - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
      - [Who are the source language producers?](#who-are-the-source-language-producers)
    - [Annotations](#annotations)
      - [Annotation process](#annotation-process)
      - [Who are the annotators?](#who-are-the-annotators)
    - [Personal and Sensitive Information](#personal-and-sensitive-information)
  - [Considerations for Using the Data](#considerations-for-using-the-data)
    - [Social Impact of Dataset](#social-impact-of-dataset)
    - [Discussion of Biases](#discussion-of-biases)
    - [Other Known Limitations](#other-known-limitations)
  - [Additional Information](#additional-information)
    - [Dataset Curators](#dataset-curators)
    - [Licensing Information](#licensing-information)
    - [Citation Information](#citation-information)
    - [Contributions](#contributions)

## Dataset Description

- **Homepage:** http://www.interactionmining.org/rico.html
- **Repository:** https://github.com/shunk031/huggingface-datasets_Rico
- **Paper (UIST2017):** https://dl.acm.org/doi/10.1145/3126594.3126651

### Dataset Summary

Rico: A Mobile App Dataset for Building Data-Driven Design Applications

### Supported Tasks and Leaderboards

[More Information Needed]

### Languages

[More Information Needed]

## Dataset Structure

### Data Instances

- UI screenshots and view hierarchies

```python
import datasets as ds

dataset = ds.load_dataset(
    path="shunk031/Rico",
    name="ui-screenshots-and-view-hierarchies",
)
```

- UI metadata

```python
import datasets as ds

dataset = ds.load_dataset(
    path="shunk031/Rico",
    name="ui-metadata",
)
```

- UI layout vectors

```python
import datasets as ds

dataset = ds.load_dataset(
    path="shunk031/Rico",
    name="ui-layout-vectors",
)
```

- Interaction traces

```python
import datasets as ds

dataset = ds.load_dataset(
    path="shunk031/Rico",
    name="interaction-traces",
)
```

- [WIP] Animations

```python
import datasets as ds

dataset = ds.load_dataset(
    path="shunk031/Rico",
    name="animations",
)
```

- Play store metadata

```python
import datasets as ds

dataset = ds.load_dataset(
    path="shunk031/Rico",
    name="play-store-metadata",
)
```

- UI screenshots and hierarchies with semantic annotations

```python
import datasets as ds

dataset = ds.load_dataset(
    path="shunk031/Rico",
    name="ui-screenshots-and-hierarchies-with-semantic-annotations",
)
```

### Data Fields

[More Information Needed]

### Data Splits

[More Information Needed]

## Dataset Creation

### Curation Rationale

[More Information Needed]

### Source Data

[More Information Needed]

#### Initial Data Collection and Normalization

[More Information Needed]

#### Who are the source language producers?

[More Information Needed]

### Annotations

[More Information Needed]

#### Annotation process

[More Information Needed]

#### Who are the annotators?

[More Information Needed]

### Personal and Sensitive Information

[More Information Needed]

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators

[More Information Needed]

### Licensing Information

[More Information Needed]

### Citation Information

```bibtex
@inproceedings{deka2017rico,
  title={Rico: A mobile app dataset for building data-driven design applications},
  author={Deka, Biplab and Huang, Zifeng and Franzen, Chad and Hibschman, Joshua and Afergan, Daniel and Li, Yang and Nichols, Jeffrey and Kumar, Ranjitha},
  booktitle={Proceedings of the 30th annual ACM symposium on user interface software and technology},
  pages={845--854},
  year={2017}
}
```

### Contributions

Thanks to [DATA DRIVEN DESIGN GROUP UNIVERSITY OF ILLINOIS AT URBANA-CHAMPAIGN](http://ranjithakumar.net/) for creating this dataset.

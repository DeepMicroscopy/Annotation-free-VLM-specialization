# Annotation-free-VLM-specialization

This repository provides source code for the COMPAYL25 workshop paper "Effortless VLM Specialization in Histopathology without Annotation" [[`paper`
](https://openreview.net/forum?id=pQDzXVeBja)] [[`blogpost`](https://deepmicroscopy.org/adapting-foundational-vlms-to-a-histopathology-task-without-any-labels-compayl25-paper-oral/)].

Abstract: Recent advances in Vision-Language Models (VLMs) in histopathology, such as CONCH and QuiltNet, have demonstrated impressive zero-shot classification capabilities across various tasks. However, their general-purpose design may lead to suboptimal performance in specific downstream applications. To address this limitation, several supervised fine-tuning methods have been proposed, which require manually labeled samples for model adaptation. This paper investigates annotation-free adaptation of VLMs through continued pretraining on domain- and task-relevant image-caption pairs extracted from existing databases. Our experiments on two VLMs, QuiltNet and CONCH, across three downstream tasks reveal that these pairs significantly enhance both zero-shot and few-shot performance. Notably, continued pretraining achieves comparable few-shot performance with larger training sizes, leveraging its task-agnostic and annotation-free nature to facilitate model adaptation for new tasks.

## Installation
This repository is built upon the [[CONCH repository](https://github.com/mahmoodlab/CONCH)], please follow the installation guidelines there and install additional packages in the [requirements.text](requirements.txt).

## Usage
* Retrieve relevant domain/task-specific image-caption pairs:
```python
python code/retrieve.py
```
* Domain/Task-specific continued pretraining:
```python
python code/finetune.py
```
* Few-shot learning (CoOp):
```python
python code/coop.py
```

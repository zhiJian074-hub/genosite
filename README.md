# GenoSite

**Pathway-Guided Multimodal Learning for Residue-Level Protein Active Site Prediction**

GenoSite estimates an active-site probability for each amino-acid residue in a protein. The framework learns from protein sequences, functional text annotations, and protein groups derived from curated pathway membership. These groups represent functional associations between proteins, not physical genomic proximity.

**Input → Output:** a protein sequence and available functional context → per-residue active-site probabilities and predicted active-site positions.

> This repository contains the research implementation. Large datasets, pretrained models, precomputed embeddings, and trained checkpoints are not included.

---

## Overview

Protein active sites are small groups of residues that are important for substrate recognition and catalytic activity. Accurate active-site prediction is useful for protein function annotation, enzyme mechanism analysis, disease-associated variant interpretation, and drug-target characterization.

Recent protein language models have improved residue-level representation learning. However, many methods still treat each protein mainly as an isolated sequence. GenoSite introduces pathway-derived functional context to provide additional biological information beyond sequence and text features.

GenoSite uses three types of input information:

* protein sequence
* functional text annotation
* pathway-derived functional context

The pathway context used in this project represents curated functional association from pathway membership. It should not be interpreted as physical genomic contiguity, chromosomal neighborhood, operon structure, or syntenic region.

---

## Repository structure

```text
genosite/
├── config/
│   └── config.yaml              # Model and training configuration
├── dataprocess/                  # Dataset preparation guidance and scripts
├── datasets/
│   └── infer_samples.tsv        # Example input data
├── pathway/                      # UniProt-to-KEGG mapping and group preparation
├── pretrained_weights/
│   └── placeholder              # Location for external model assets
├── utils/                        # Shared utilities
├── environment.yml              # Reference Conda environment
├── requirements.txt             # Partial dependency list
├── inference.py                  # Research inference script
├── model.py                      # GenoSite model
└── train.py                      # Training and evaluation
```


---

## Environment

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate genosite
```

Alternatively, install dependencies with:

```bash
pip install -r requirements.txt
```

Please make sure that PyTorch, CUDA, ESM, and other dependencies are compatible with your local environment.

---

## Data and Checkpoints

This repository does not include:

* full datasets
* pretrained weights
* generated embeddings
* pathway-context intermediate files
* trained checkpoints
* experiment outputs

---

## Training

Run the main training script:

```bash
python train.py --config config/config.yaml
```

---

## Inference

Run inference with:

```bash
python inference.py
```

For embedding-based inference:


Please make sure that the required model checkpoint and input data are prepared before running inference.


## Citation

If you find this repository useful, please consider citing the related MMSite work and this repository.

```bibtex
@misc{genosite,
  title  = {GenoSite: Pathway-Guided Multimodal Learning for Residue-Level Protein Active Site Prediction},
  author = {zhiJian074-hub},
  year   = {2026},
  url    = {https://github.com/zhiJian074-hub/genosite}
}
```

---

## License

This repository is released for academic research purposes.

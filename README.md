# MAF_AFP
## Introduction
MafAFP is a biophysically motivated multi-attribute modulation framework for AFP activity prediction. MafAFP integrates a protein language model with a multi-attribute biophysically motivated antifungal mechanism feature encoder that captures key antifungal mechanism cues (e.g., insertion propensity, amphipathic helicity, and membrane affinity ) across multiple sliding-window scales. These mechanism-relevant signals are then injected into PLM embeddings through Feature-wise Linear Modulation (FiLM) and gated residual fusion, enabling adaptive, mechanism-guided representation learning. Comprehensive benchmarks show that MafAFP achieves competitive predictive performance while offering improved mechanistic interpretability and more robust cross-datasets generalization.
## Environment
* python 3.10
* Use conda to creat new environment
* conda env create -f environment.yaml
## Usage
* Please download the relevant files of the esmc_600, pre-trained model on Hugging Face and put them in the created MafExtractor/predict_model/model/weights folder.
* run Maf_train.py and MafAFP_train.py to train MafExtractor and AFP predict model

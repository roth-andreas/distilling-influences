# Repository

This repository contains the implementation for "Influence Matching for Reduced Prediction Churn in Knowledge Distillation for Node Classification". Two files need to be executed: gnn.py and train_teacher.py, for training student and teacher models. The run.sh script is used to execute the training process with default settings.
## Dependencies

This code requires the following dependencies to be installed:

* Python > 3
* PyTorch > 2.0
* PyTorch Geometric >= 2.3

## Usage

To use this code, you can execute the run.sh script. This will train all models as presented in the paper. Alternatively, you can run teacher and student models directly.
## File descriptions
### gnn.py

This file contains the implementation of all student models.
### train_teacher.py

This file contains the training code for the teacher model.
### run.sh

This is a shell script that directly runs all models. To run this file, please provide a conda environment in the first line of the file.
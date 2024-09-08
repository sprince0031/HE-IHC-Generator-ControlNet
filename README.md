# Generating HER2 Immunohistochemical Images from H&E Stains using Stable Diffusion and ControlNet

This is the companion code repository for my masters thesis dissertation. The below listed files are the ones that have been written and used for the implementation of the research in the thesis.

## Files used
1. [IHC_Dataset.py](/IHC_Dataset.py) - This file is used to create the dataset object for the HE-IHC image pairs from the BCI dataset. It uses the `torch.utils.data.Dataset` class to create the dataset.
2. [SDUtils.py](/SDUtils.py) - This file contains the utility functions that are used in the Stable Diffusion model. It contains helper functions to help form the dataset, plot the image patches, etc.
3. [tool_add_control_sd21.py](/tool_add_control_sd21.py) - This is the utility script provided by the original ControlNet repository that creates a trainable copy of the Stable Diffusion 2.1 network with all of its pretrained weights.
4. [trainer.py](/trainer.py) - The main training script derived from the provided example script, [tutorial_train.py](/tutorial_train.py) which uses PyTorch Lightning to start the training loop.
5. [testerv1.py](/testerv1.py) - The testing script to test the finetuned model on the test dataset without the use of prompts.
6. [testerWithPrompt.py](/testerWithPrompt.py) - The testing script to test the finetuned model on the test dataset WITH the use of prompts.
7. [dissertation_main_results_notebook.ipynb](/dissertation_main_results_notebook.ipynb) - The Jupyter notebook that contains the code to generate the results and plots for the dissertation.

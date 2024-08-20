![Raygun blasting a protein, shrinking its size.](logo-tiny.png)
# Raygun: template-based protein design tool

Raygun is a new approach to protein design. Unlike de novo design tools
that generate a protein from scratch, Raygun allows users to take an
existing protein as template and modify it by introducing insertions,
deletions and substitutions. Our analyses showed that the modified
proteins significantly retained structural and functional properties of
the original template protein. We anticipate Raygun to be valuable in a variety 
of applications related to protein miniaturization, property optimization and so on.

## Preprint

**Devkota, K., Shonai, D., Mao, J., Soderling, S. H., & Singh, R.
(2024). Miniaturizing, Modifying, and Augmenting Nature's Proteins with
Raygun. bioRxiv, 2024-08.** [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2024.08.13.607858)


## Introduction

Raygun is a novel protein design framework that allows for
miniaturization, magnification and modification of any template
proteins. It lets the user select any protein as template and generates
structurally (and therefore, functionally) similar samples, while giving
full control over the lengths of the generated sequences.

**How to use Raygun:** Input a protein sequence, specify a target length
and a noise parameter. Raygun will use those information to efficiently
generate samples (< 1 sec/sample on a GPU). The users have absolute
control over the length of the target protein.

**How Raygun works** Raygun is an autoencoder-based design which
represents any protein as a 64,000-dimensional Multivariate Normal
Distribution. The Raygun decoder has the ability to accurately map this
fixed-length representation back to the variable length space of the
user's specifications.


## (Opinionated) Guidance on how best to use Raygun

The thing that Raygun seems to do well is to preserve the general structure and return a candidate that will likely fold in vitro and retain *some* functional properties. However, the more of a handle you have on predicting/optimizing function, the more useful Raygun can be for you. Raygun is very fast, so you can use it to generate lots of candidates and filter them down. We already do that as a first pass, with pseudo-loglikelihood. A good next filter might be using HMMER to prioritize well-preserved PFAM domains. If there's an additional filter you can apply (e.g., thermostability) that might help. However, be aware that many DMS datasets used for property-prediction are probably suitable more for assessing substitutions than indels. That's where having an experimental pipeline to test a few initial candidates and optimize them in the lab would help.  

### Requirements

Raygun has a few package requirements: `numpy`,
`pandas`, `fair-esm`, `pyyaml`, `h5py`, `einops` and `torch`
(the version suitable for your GPU). We verified that our model works on
A100 and A6000 GPUs, for the following specifications:

-   fair-esm=2.0.0
-   numpy=1.26.4
-   pandas=2.1.4
-   pytorch=2.1.1 (py3.11_cuda12.1_cudnn8.9.2_0)

### From source repository

Users can install Raygun directly from source by cloning the github repo
<https://github.com/rohitsinghlab/raygun> and installing the package
through pip.

``` bash
git clone https://github.com/rohitsinghlab/raygun
cd raygun
pip install . #note that the code will be copied to the environment's packages directory, so your localdir changes will not be reflected unless you reinstall
```

### Using pip

Alternately, users can install raygun from the pip repository

``` bash
pip install raygun
```

## Quick start

Raygun provides users with two command-line programs for training the
model and fine-tuning/generating protein samples. These are described
below

### Generating samples

After the raygun package has been installed, we can use it to generate
samples using the `raygun-sample` command. This method will
also fine-tune the model.

We strongly recommend that the user first fine-tune the model on the
target sequence or a set of related sequences.

`raygun-sample` can be invoked in bash in the following way:

``` bash
raygun-sample --config <YAML configuration file>
```

We have provided YAML configuration files related to lacZ sampling in the github repository folder `example-configs/lacZ`:

-   Quick Start: `generate-sample-lacZ-v1.yaml`
    fine-tunes on just one lacZ template sequence, and then
    generates.
-   Full Example: `generate-sample-lacZ-v1.yaml`
    fine-tunes on 20 lacZ sequences from the relevant PFAM domain,
    and then generates.

Below we show `v1`

``` YAML
## This YAML file specifies all the parameters for using Raygun. ##
##  At start, we suggest focusing only on parameters in Sections 1 and 2.  

###### Section 1: GPU, INPUT and OUTPUT Locations ########
device: 0  # CUDA device
## template FASTA file
templatefasta: "example-configs/lacZ/lacZ-template.fasta" 


## FINE-TUNING ##

## We strongly recommend starting from our pre-trained
## model and fine-tune it for your sequences.

## First time fine-tuning (or to overwrite previous fine-tune). Comment these lines if reusing fine-tuned model. 
finetune: true  # will overwrite the existing models in model folder if it exists
finetunetrain: "example-configs/lacZ/lacZ-template.fasta" # a single fasta file containing 1 or more sequences you want to fine-tune the decoder on
finetuned_model_loc: "lacZ-finetuned"  # folder where models are saved. Will be created if it doesn't exist

## Uncomment lines below to reuse fine-tuned model. 
# finetune: false
# finetuned_model_checkpoint: "lacZ-model/epoch_50.sav" 


## OUTPUT LOCATION ##

## output folder. Will be created if does not exist. Files may be overwritten if names clash
output_file_identifier: "lacZ"  # this will be a substring in all output files produced
sample_out_folder: "lacZ-samples"


###### Section 2: GENERATION PARAMETERS ########

# how many samples you want. This will be the count after the PLL filtering
num_raygun_samples_to_generate: 20

## how much pseudo log-likelihood filtering to do.
## value=0 means no filtering, 0.9 means keep best 10% of hits by PLL
## Raygun will actually generate <num_raygun_samples_to_generate>*(1/<filter_ratio_with_pll>) entries, storing them in a file with "unfiltered" in its name
##  It'll then filter them by PLL and store the <num_raygun_samples_to_generate> sequences in a file with "filtered" in its name
filter_ratio_with_pll: 0.5

## target lengths: a json file containing the target length range you want for each template
##
##  the format of the json file is: { "<fastaid>": [minlen, maxlen], ... }
##   here's an example: {  "sp|P00722|BGAL_ECOLI": [900, 950] }.
##   In this case, a total of <num_raygun_samples_to_generate> will be generated across 900-950 length
##  to specify a single target-length, you can set minlen=maxlen (e.g. [900,900])

lengthinfo: "example-configs/lacZ/leninfo-lacZ.json"

## noiseratio is a number >= 0. At 0, minimal substitutions will be introduced. If you go over 2.25, expect >50% substitution rate
## for most applications, we recommend noiseratio = 0.5 and randomize_noise = true 
noiseratio: 0.5
randomize_noise: true  # if true, the actual noise for any sample will actually be sampled from uniform(0, <noiseratio>)


###### Section 3: OTHER PARAMETERS ########
## you can ignore these for now

finetune_epoch: 50
finetune_save_every: 50
finetune_lr: 0.0001
minallowedlength: 55 # minimum length of template protein
usereconstructionloss: true
usereplicateloss: true
usecrossentropyloss: true
reconstructionlossratio: 1
replicatelossratio: 1
crossentropylossratio: 1
maxlength: 1000 
saveoptimizerstate: false
```

Note that `raygun-sample` gives users the option to perform
finetuning on the pretrained model. So, only using
`raygun-sample` satisfies majority of end-user requirements

### Training the model

If the goal is to pre-train the model from scratch, we suggest using the
`raygun-train` command. It can be invoked as:

``` bash
raygun-train --config <YAML configuration file>
```

We also provide the configuration file for training the lacZ-train model
in the `example-configs/lacZ` folder.

``` YAML
## This YAML file specifies all the parameters for finetuning Raygun, or training it from scratch ##
##  At start, we suggest focusing only on parameters in Sections 1 and 2.  
###### Section 1: GPU, INPUT and OUTPUT Locations ########
# your cuda device. If you only have one, 0 is likely the default
device: 0

## INPUT ##

# the input sequence(s) for training or finetuning. Requires the training fasta file. If the validation fasta not specified, the training epoch will not perform the validation step
trainfasta: "example-configs/lacZ/lacZ-selected-family.fasta"
# validfasta: "example-configs/lacZ/lacZ-selected-family.fasta"

# embedding location. If specified, the ESM-2 outputs will be saved at this location. If set to null, the embeddings are not saved
esm2_embedding_saveloc: null

# folder where the output model is to be saved. REQUIRED
output_model_loc: "lacZ-trained"
# if the `checkpoint` is specified, the training/finetuning will begin from this checkpoint. If not provided, the program will use the pretrain model
# checkpoint: "bgal-model/epoch_5.sav"

# Set this to true if the goal is finetuning. Finetuning freezes the encoder parameters only updating the Raygun decoder
# For training from scratch, set `finetune: false`. Default: false
finetune: false

## Total number of epochs to train/finetune, and the learning rate
epoch: 50
lr: 0.0001
# Default: 1, model is saved at every multiples of this parameter
save_every: 50

###### Section 3: OTHER PARAMETERS ########
## you can ignore these for now
usereconstructionloss: true
usereplicateloss: true
usecrossentropyloss: true
reconstructionlossratio: 1
replicatelossratio: 1
crossentropylossratio: 1
maxlength: 1000
minallowedlength: 55
clip: 0.0001
saveoptimizerstate: false
```


## License
Everything in this repository is licensed under the CC-BY-NC license. In addition to the terms of this license, we grant the following rights:

 - Employees of governmental, non-profit, or charitable institutions (including most academic researchers) are permitted to use Raygun as part of a workflow that results in commercial products or services. For example, if you are an academic who creates a molecule using Raygun and wish to commercialize it, you are welcome to do so.

 - For-profit organizations are allowed to use Raygun for public-domain outputs, such as publications or preprints. Additionally, these organizations are granted a 90-day trial license for internal evaluation purposes. Continued use beyond the trial period or for any commercial activities will require a separate license agreement.


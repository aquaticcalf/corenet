> [!important]  
> This is my fork of [`apple/corenet`](https://github.com/apple/corenet).  Any changes made here are just me playing around with the code and not intended to contribute to the original project. They may be experimental, broken, or incomplete, so use at your own risk!
>
> Additionally, please note that this fork is not affiliated with Apple Inc. or any of its subsidiaries. I am not an Apple employee, and any modifications made here do not reflect the opinions or policies of Apple. This is a personal project, and I am solely responsible for its content.

# CoreNet: A library for training deep neural networks

CoreNet is a deep neural network toolkit that allows researchers and engineers to train standard and novel small and large-scale models for variety of tasks, including foundation models (e.g., CLIP and LLM), object classification, object detection, and semantic segmentation.

## Table of contents

   * [Installation](#installation)
   * [Directory Structure](#directory-structure)
   * [Relationship with CVNets](#relationship-with-cvnets)

## Installation

You will need Git LFS (instructions below) to run tests and Jupyter notebooks 
([instructions](https://jupyter.org/install)) in this repository,
and to contribute to it so we recommend that you install and activate it first.

On Linux we recommend to use Python 3.10+ and PyTorch (version >= v2.1.0), on
macOS system Python 3.9+ should be sufficient.

Note that the optional dependencies listed below are required if you'd like to
make contributions and/or run tests.

For Linux (substitute `apt` for your package manager):

```bash
sudo apt install git-lfs

git clone git@github.com:aquaticcalf/corenet.git
cd corenet
git lfs install
git lfs pull
# The following venv command is optional, but recommended. Alternatively, you can create and activate a conda environment.
python3 -m venv venv && source venv/bin/activate
python3 -m pip install --editable .
```

To install optional dependencies for audio and video processing:

```bash
sudo apt install libsox-dev ffmpeg
```

For macOS, assuming you use Homebrew:

```bash
brew install git-lfs

git clone git@github.com:aquaticcalf/corenet.git
cd corenet
cd \$(pwd -P)  # See the note below.
git lfs install
git lfs pull
# The following venv command is optional, but recommended. Alternatively, you can create and activate a conda environment.
python3 -m venv venv && source venv/bin/activate
python3 -m pip install --editable .
```

To install optional dependencies for audio and video processing:

```bash
brew install sox ffmpeg
```

Note that on macOS the file system is case insensitive, and case sensitivity
can cause issues with Git. You should access the repository on disk as if the
path were case sensitive, i.e. with the same capitalization as you see when you
list the directories `ls`. You can switch to such a path with the `cd $(pwd -P)`
command.


## Directory Structure

This section provides quick access and a brief description for important CoreNet directories.

<table>
<thead>
<tr>
<th> Description </th>
<th> Quick Access </th>
</tr>
</thead>
<tbody>
<!-- Row boilerplate (copy-paste the following commented snippet for adding a new row to the table.)
<tr> <td> <h3> title </h3> 
description
</td> <td> <pre>
folders
</pre> </td> </tr>
-->
<tr> <td> <h3> Getting Started </h3> 
Working with the examples is an easy way to get started with CoreNet. 
</td> <td> <pre>
└── tutorials
    ├── <a href="tutorials/train_a_new_model_on_a_new_dataset_from_scratch.ipynb">train_a_new_model_on_a_new_dataset_from_scratch.ipynb</a>
    ├── <a href="tutorials/guide_slurm_and_multi_node_training.md">guide_slurm_and_multi_node_training.md</a>
    ├── <a href="tutorials/clip.ipynb">clip.ipynb</a>
    ├── <a href="tutorials/semantic_segmentation.ipynb">semantic_segmentation.ipynb</a>
    └── <a href="tutorials/object_detection.ipynb">object_detection.ipynb</a>
</pre> </td> </tr>


<tr> <td> <h3> Training Recipes </h3>
CoreNet provides reproducible training recipes, in addition to the pretrained model 
weights and checkpoints for the publications that are listed in <code>projects/</code> directory.

Publication project directories generally contain the following contents:

* `README.md` provides documentation, links to the pretrained weights, and citations.
* `<task_name>/<model_name>.yaml` provides configuration for reproducing the trainings and evaluations.
</td> <td> <pre>
└── projects
    ├── <a href="projects/byteformer">byteformer</a>
    ├── <a href="projects/catlip">catlip</a> (*)
    ├── <a href="projects/clip">clip</a>
    ├── <a href="projects/fastvit">fastvit</a>
    ├── <a href="projects/mobilenet_v1">mobilenet_v1</a>
    ├── <a href="projects/mobilenet_v2">mobilenet_v2</a>
    ├── <a href="projects/mobilenet_v3">mobilenet_v3</a>
    ├── <a href="projects/mobileone">mobileone</a>
    ├── <a href="projects/mobilevit">mobilevit</a>
    ├── <a href="projects/mobilevit_v2">mobilevit_v2</a>
    ├── <a href="projects/openelm">openelm</a> (*)
    ├── <a href="projects/range_augment">range_augment</a>
    ├── <a href="projects/resnet">resnet</a>
    └── <a href="projects/vit">vit</a>
<br>
(*) Newly released.
</pre> </td> </tr>


<tr> <td> <h3> MLX Examples </h3>
MLX examples demonstrate how to run CoreNet models efficiently on Apple Silicon.
Please find further information in the <code>README.md</code> file within the corresponding example directory.

</td> <td> <pre>
└──mlx_example
    ├── <a href="mlx_examples/clip">clip</a>
    └── <a href="mlx_examples/open_elm">open_elm</a>
</pre> </td> </tr>


<tr> <td> <h3> Model Implementations </h3> 
Models are organized by tasks (e.g. "classification"). You can find all model implementations for each
task in the corresponding task folder. 

Each model class is decorated by a 
`@MODEL_REGISTRY.register(name="<model_name>", type="<task_name>")` decorator. 
To use a model class in CoreNet training or evaluation,
assign `models.<task_name>.name = <model_name>` in the YAML configuration.

</td> <td> <pre>
└── corenet
    └── modeling
        └── <a href="corenet/modeling/models">models</a>
            ├── <a href="corenet/modeling/models/audio_classification">audio_classification</a>
            ├── <a href="corenet/modeling/models/classification">classification</a>
            ├── <a href="corenet/modeling/models/detection">detection</a>
            ├── <a href="corenet/modeling/models/language_modeling">language_modeling</a>
            ├── <a href="corenet/modeling/models/multi_modal_img_text">multi_modal_img_text</a>
            └── <a href="corenet/modeling/models/segmentation">segmentation</a>
</pre> </td> </tr>


<tr> <td> <h3> Datasets </h3> 
Similarly to the models, datasets are also categorized by tasks.
</td> <td> <pre>
└── corenet
    └── data
        └── <a href="corenet/data/datasets">datasets</a>
            ├── <a href="corenet/data/datasets/audio_classification">audio_classification</a>
            ├── <a href="corenet/data/datasets/classification">classification</a>
            ├── <a href="corenet/data/datasets/detection">detection</a>
            ├── <a href="corenet/data/datasets/language_modeling">language_modeling</a>
            ├── <a href="corenet/data/datasets/multi_modal_img_text">multi_modal_img_text</a>
            └── <a href="corenet/data/datasets/segmentation">segmentation</a>
</pre> </td> </tr>


<tr> <td> <h3> Other key directories </h3> 
In this section, we have highlighted the rest of the key directories that implement 
classes corresponding to the names that are referenced in the YAML configurations.
</td> <td> <pre>
└── corenet
    ├── <a href="corenet/loss_fn">loss_fn</a>
    ├── <a href="corenet/metrics">metrics</a>
    ├── <a href="corenet/optims">optims</a>
    │   └── <a href="corenet/optims/scheduler">scheduler</a>
    ├── <a href="corenet/train_eval_pipelines">train_eval_pipelines</a>
    ├── <a href="corenet/data">data</a>
    │   ├── <a href="corenet/data/collate_fns">collate_fns</a>
    │   ├── <a href="corenet/data/sampler">sampler</a>
    │   ├── <a href="corenet/data/text_tokenizer">text_tokenizer</a>
    │   ├── <a href="corenet/data/transforms">transforms</a>
    │   └── <a href="corenet/data/video_reader">video_reader</a>
    └── <a href="corenet/modeling">modeling</a>
        ├── <a href="corenet/modeling/layers">layers</a>
        ├── <a href="corenet/modeling/modules">modules</a>
        ├── <a href="corenet/modeling/neural_augmentor">neural_augmentor</a>
        └── <a href="corenet/modeling/text_encoders">text_encoders</a>
</pre> </td> </tr>

</tbody>
</table>

## Relationship with CVNets

CoreNet evolved from CVNets, to encompass a broader range of applications beyond computer vision. Its expansion facilitated the training of foundational models, including LLMs.

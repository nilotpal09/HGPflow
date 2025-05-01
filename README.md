This repository contains the code used in the study, described in [HGPflow: Extending Hypergraph Particle Flow to Collider Event Reconstruction](https://arxiv.org/abs/2410.23236v1)

---

## Training

Recommended training strategy is to train in two stages. Stage one learns the incidence matrices and stage two learns the corrections. Prior to running stage two, we need to run inference on stage one. This allows for having more educated guess for the hyperparameters in stage two.

To train stage one, we can run the following -

```
python -m hgpflow_v2.train -cv path/to/var_config.yml -cms1 path/to/stage1_config.yml -ct path/to/train_config.yml
```

It also has the following optional flags:
```
--exp_key (-ekey)
--debug_mode (-d)
--precision (-p)
--gpu (-g)
```

Next step will be to run inference on stage one. This will generate the training data for stage two.  

```
python -m hgpflow_v2.hyperedge_data_prep -i path/to/inference_stage1.yml
```

Then to train stage two, we can run the following (same command as above, but we replace `cms1` with `cms2`)-


```
python -m hgpflow_v2.train -cv path/to/var_config.yml -cms2 path/to/stage2_config.yml -ct path/to/train_config.yml
```

It also has thee same optional flags as above. (We use the same training script for both stages, so the flags are the same)

We need to provide the stage one model config (`cms1`) and the ckeckpoint path in the config for stage two. This is because the output of the stage two model is the full model, which includes the stage one model. This is just to make the inference easier with only one model. The stage one model is frozen during training of stage two.

---

## Inference (full pipeline)

To run inference, we need four things. The variable config, the stage one model config, the sage tow model config, and the final checkpoint (coming from stage2 trainings). We can put the paths to these four in a yaml file and run the following -

```
python -m hgpflow_v2.eval -i path/to/inference.yml
```

---

## Performance Plots

plots can be made with the notebooks - [notebooks/clic/clic_performance.ipynb]() and [notebooks/cocoa/cocoa_performance.ipynb]()

---

## Trained models

The checkpoints used to make the plots in the paper are available in `saved_checkpoints` directory

---

## Modified MLPF code

The modified MLPF code to train COCOA, and CLIC (correcting the truth definition) are available [here](https://github.com/annaivina/particleflow-fork)

# Direct Consistency Optimization (DCO) Reimplementation

This is a reimplementation of the Direct Consistency Optimization (DCO) paper.

I’ve made some additional modifications to the DCO loss functions, which you can find in the `training/training_loop_` directory. There are several different versions of the loss function there.

> **Note:** The most successful version is in `training_loop_fixed.py`. To run this code, rename the file to `training_loop.py`. I apologize for the messy implementation—I’ll tidy it up as soon as possible.

The idea behind the loss function in the `training_loop_fixed.py` file is to fully utilize the original DPO loss function (you can find this formula in the Diffusion for DPO paper).

>  Your data will be set as personalized data, and the output of the fine-tuning model, before `n` steps, will be treated as the loss data.  
> *(This `n` is a hyperparameter, and experimentally, values between 50 and 100 worked best.)*

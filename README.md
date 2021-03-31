This repo is now effectively a fork of https://github.com/avikj/L4DC-MPC-OCD that extracts the environment from interact_drive, plus some additonal building to get a backwards compatible version of the environment from "Active Preference-Based Learning of Reward Functions" by Sadigh et. al. with different features. Most of the credit to Lawrence Chan and Avik Jain.

The relevant portions of the original readme follow.

There are two sets of legacy objects. The first live in the `legacy/` folder and have the structure from many of Sadigh's original papers, left for compatabiliy with active-irl algorithms. The second are classes named `LegacySomething` e.g. `LegacyPlannerCar` which are intended for use with the L4DC-MPC-OCD code and provide gradients, but replicate the dynamics of the original environment more accurately.
# TODO(joschnei): Detailed usage instructions.

---

## Requirements
```
python>=3.6
```

## Installation

```
pip install -e .
```


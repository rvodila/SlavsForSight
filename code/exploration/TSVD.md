# TVSD

## THINGS Ventral stream Spiking Dataset (TVSD)

![logo](https://gin.g-node.org/paolo_papale/TVSD/raw/master/_code/code_utils_v2/_figs/logo1.png)

###### 
Large-scale electrophysiological recordings from V1, V4 and IT in two macaques in response to ~22k images from the THINGS image database. 

**Paolo Papale, Feng Wang, Matthew W. Self and Pieter R. Roelfsema**

*Dept. of Vision and Cognition, Netherlands Institute for Neuroscience (KNAW), 1105 BA Amsterdam (NL).*


N.B. The THINGS stimuli from Martin Hebart (et al.) are not provided, but you can download them on [things-initiative.org](http://things-initiative.org)

N.B.B. We're adding answers to the most common issues and questions here: https://gin.g-node.org/paolo_papale/TVSD/issues (see "closed issues" as well).

## Info

A full description of the dataset is provided in the Neuron paper. A few additional things are needed to be able to work on the data. 

### RAW

Data for each monkey is provided in a folder, containing both the RAW and MUA data. The RAW data is subdived in different days of recordings, and individual blocks/runs of ~20 minutes of lenght. We provide the MATLAB code to extract the MUA out of RAW data, aggregate all the trials across blocks and days, normalize it, filter and chunck it for model training. These scripts can be easily changed to extract LFP, or to aggregate the RAW data, or look into not-completed images.

These scripts can be found in "_code" and are (in sequence):
- extract_MUA_v2.m (for monkeyF) and extraxt_MUA_v2_N.m (for monkeyN - we used 2 different recording systems)
- collect_MUA_v2.m
- norm_MUA.m
- export_MUA.m

The MUA data is provided both un-normalized ("THINGS_MUA_trials.mat") and normalized and averaged in time-windows ("THINGS_normMUA.mat")

### MUA
#### (if interested in the full time-course or decoding)

THINGS_MUA_trials.mat contains:

- ALLMAT: has info on what was shown in each stimulus (rows). The columns are: [#trial_idx #train_idx #test_idx #rep #count #correct #day]. Most importat stuff are "train_idx" and "test_idx" which tells you the ID of the stimulus. If "train_idx" for a specific row is 0, that means that a test image was shown, and the other way around. "count" is the position of the stimulus in the sequence of 4 images that were showed to the monkeys.  "rep" (i.e. repetitions) was reshuffled by randomization so it's practically useless. "trial_idx" is linearly adding up. Only correct trials are included! "day" is just the day of that session.
- ALLMUA: rows correspond to stimuli. The columns are: [#electrode #trial_idx #time-points].
- tb: time, in ms, w.r.t. the stimulus onset, correspoing to the elements in "time-points" in "ALLMUA".

N.B.: please keep in mind that the order of channels in ALLMUA does not follow the real arrangment of arrays and ROIs! Instead, it follows the arrangment of the recording system that is shuffled by the design of the headstages. Please see https://gin.g-node.org/paolo_papale/TVSD/issues/2 for more details on how to (very quickly) re-map the data in the correct order!


### normalized MUA
#### (if interested in modeling and tuning)

THINGS_normMUA.mat contains:

- SNR: the signal-to-noise ratio, computed for each day of recordings [#electrodes #days] (described in the paper)
- SNR_max: the SNR_max (described in the paper)
- lats: latency of the onset of the (mean) response, computed for each day of recordings [#electrodes #days] (described in the paper)
- reliab: the reliability of each electrode - the mean reliability described in the paper is the just the mean of reliab across combinations of trials
- oracle: the oracle correlation (described in the paper)
- train_MUA: averaged and normalized response to each train stimulus, already sorted to match "things_imgs.mat" (see below) [#electrode #stimuli] (described in the paper)
- test_MUA: averaged and normalized response to each test stimulus, already sorted to match "things_imgs.mat" (see below) [#electrode #stimuli] (described in the paper)
- test_MUA_reps: normalized response to each test stimulus and repetition, already sorted to match "things_imgs.mat" (see below) [#electrode #stimuli] (described in the paper)
- tb: time, in ms, w.r.t. the stimulus onset, correspoing to the elements in "time-points" in "ALLMUA".


### Stimuli

The scripts and data rely on logfiles hosted in "_logs". There, you can also find "things_imgs.mat" that is required to associate each stimulus from the THINGS initiative database to the specific trial (see below). things_imgs.mat contains:

- train_imgs: has info on the stimulus ID, e.g. field 1 corresponds to stim 1 "train_idx" in ALLMAT. 
- test_imgs: has info on the stimulus ID, e.g. field 1 corresponds to stim 1 "test_idx" in ALLMAT.

N.B. the normalized data is already sorted according to the order of images in "train_imgs" and "test_imgs" from "things_imgs.mat"

### Code

In addition to the scripts mentioned, we provide the Matlab APIs from Blackrock Neurotech, the same version used for the paper. Also, we provide a few util functions that are called by the main scripts or can be used to plot some of the results. Finally, we provide the Python code for the MEIs in "lucent-things", based on the lucent viz library [GitHub link](https://github.com/greentfrapp/lucent)



## Downloading the data
### Using gin

Create an account on gin and download the gin client as described [here](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Setup).
On your computer, log in using:

```sh
gin login
```

Clone the repository using:

```sh
gin get paolo_papale/TVSD
```
The cloning step can take a long time, due to the large amount of individual files. Please be patient.

Large data files will not be downloaded automatically, they will appear as git-annex links instead. We recommend downloading only the files you need, since the entire dataset is large. To get the contents of a certain file:

```sh
gin get-content <filename>
```

Downloaded large files will be read-only. You might want to unlock the files using:

```sh
gin unlock <filename>
```

To remove the contents of a large file again, use:

```sh
gin remove-content <filename>
```

Detailed description of the gin client can be found at the [gin wiki](https://gin.g-node.org/G-Node/Info/wiki/). See the gin [usage tutorial](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Usage+Tutorial) for advanced features.

### Using the web browser

Download the files you want by clicking download in the gin web interface.

If you are interested in modeling/tuning, you can download the normalized MUA for each monkey by clicking here:

monkey N: https://gin.g-node.org/paolo_papale/TVSD/raw/master/monkeyN/THINGS_normMUA.mat

monkey F: https://gin.g-node.org/paolo_papale/TVSD/raw/master/monkeyF/THINGS_normMUA.mat

## Citation policy

Cite this work by citing the original publication.

## Contact information

Don't use our institutional emails for questions about the TVSD, instead you can reach us at things [dot] tvsd [at] gmail [dot] com

## License
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />The data and metadata in this work are licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

Python (v. 2.7) and Matlab (v. 2019b) code in this repository are licensed under the same license, with the following exceptions:

The matlab-based NPMK package provided within this repository is re-distributed under the BSD 3-clause license, in compliance with the original licensing terms.

The python-based lucent library provided within this repository is re-distributed under the Apache License 2.0 license, in compliance with the original licensing terms.

The python-based models under lucent provided within this repository are re-distributed under the MIT License license, in compliance with the original licensing terms.
# Anxiety
Development and Experiment Record for the CPNE Anxiety Project

### System Requirements

- Must be able to run/build a Singularity container https://docs.sylabs.io/guides/latest/user-guide/. Usually this is easiest on a system running linux or on the windows subsystem for linux.
- This code was developed and tested on a high performance computing environment running a singularity container with an ubuntu base. 
- While model training can be accelerated with GPU hardware, this is not strictly required. Due to the size of the datasets, 32 GB of memory or more is recommended.

### Installation Guide
- Follow instructions for installing Singularity on your system including intalling Go https://docs.sylabs.io/guides/latest/user-guide/quick_start.html#quick-installation-steps.
- Using the provided cpne_anxiety.def file, run "sudo singularity build cpne_anxiety.simg cpne_anxiety.def"
- initialize the singularity environment. For simple use cases, use "singularity run cpne_anxiety.simg -i"
- Navigate to /Anxiety/lpne/ and run `pip install .`
- All dependencies should now be installed
- It is expected that installation time should take less than 30 minutes on a desktop computer.

### Demo

A demo for projecting new datasets is shown in the Anxiety_Network_Projection folder as a jupyter notebook. This pipeline demonstrates how to generate network projections of the anxiety networks which was done to generate all anxiety network scores. The expected output is saved as an html file in the folder. Data for running the demo will be provided with the data release prior to publication.

### Instructions for Use
All data is saved to excel or csv format sheets on which various statistical software can be freely applied.

### Reproduction
All figure statistics can be found in /Anxiety/FullDataWork/Paper_Statistics/Paper_Stats/ with corresponding names for each figure. All data for generating the figures is found in the github. Some stats in the manuscript are instead found in /Anxiety/FullDataWork/Projections/.

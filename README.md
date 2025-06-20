# Deep Brain Stimulation: Classification of STN-DBS ON and OFF states

For the classification of the STN-DBS ON/OFF states,
the extracted feature maps were organised into a vectorized format reshaping
the 3D data into a 1D vector.
A mask was later applied to remove the zero values surrounding the brain so
that each element in the vectors will represent a specific voxel of the
corresponding connectivity map.
For each measure, nine classification algorithms were implemented.
The default parameters were used for all cases.

---

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/kiakoudimi/NeuroDBS.git
    ```

2. **Create a conda environment**:

    ```bash
    conda create -n neuroDBS python=3.10
    conda activate neuroDBS
    ```

3. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Run the Script

After setting up your environment and installing dependencies, you can run the main scripts as follows:

1. **Run via SLURM**:

   ```bash
   cd ../NeuroDBS/scripts/
   sbatch main_test_job.sh
   ```

2. **Run directly**:

   ```bash
   cd ../NeuroDBS/scripts/
   python main_test.py
   ```

The same steps apply for running the `main_train.sh` and `main_train.py`

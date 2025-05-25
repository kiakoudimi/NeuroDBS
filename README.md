# Deep Brain Stimulation: CLassification of STN-DBS ON and OFF states
For the classification of the STN-DBS ON/OFF states, the extracted feature maps were organised into a vectorized format reshaping the 3D data into a 1D vector. A mask was later applied to remove the zero values surrounding the brain so that each element in the vectors will represent a specific voxel of the corresponding connectivity map. For each measure, nine classification algorithms were implemented. The default parameters were used for all cases.

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/kiakoudimi/NeuroDBS.git

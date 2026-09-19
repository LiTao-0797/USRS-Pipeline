# USRS-Pipeline <br/>
Pipeline for tree point cloud registration and segmentation. <br/>

## Installation <br/>
OS: Windwos 10 or 11 <br/>
RAM: 64GB or above <br/>
IDE: Spyder (in Anaconda) <br/>
Environment: open3d 0.19.0, numpy 1.26.3, gtsam 4.2a9, scipy 1.13.1, tqdm 4.67.1, plyfile 1.1.2, jakteristics 0.6.2, networkx 3.2.1, matplotlib 3.9.4, CSF 1.0.20.0, opencv-python 4.11.0.86, sklearn 1.6.1, laspy 2.5.4 <br/>

## Usage <br/>
### EVO dataset <br/>
1. First, download the dataset from https://drive.google.com/drive/u/0/folders/13oNJS5qsoNYkJzNBXBWpAf0FpkaFOr-H (Casseau et al.), then create a 'datasets' folder in the root directory of the code, and put the downloaded 'evo_example_dataset' dataset into it. <br/>
2. Open the file "step1_pre_data.py" in Spyder, then click "Run" to start the data preprocessing. Please note to check the dataset storage path. <br/>
3. After completing the data preprocessing, open the file "step2_local_registration.py" in Spyder, then click "Run" to start local point cloud registration. <br/>
4. After completing the local registration, open the file "step3_global_optimization.py" in Spyder, then click "Run" to start global point cloud optimization. <br/>
5. Finaly, open the file "step4_seed_guided_growth_segmentation.py" in Spyder, then click "Run" to start individual tree segmentation based on stem seed. <br/>

Thanks to Casseau et al. for providing the dataset (https://doi.org/10.1109/IROS58592.2024.10802448). <br/>
### Wytham woods dataset <br/>
1. First, download the 'wytham_vox0.1.laz' from https://data.goettingen-research-online.de/dataset.xhtml?persistentId=doi:10.25625/QUTUWU (Henrich et al.), and put the downloaded 'wytham_vox0.1.laz' into 'datasets' folder. <br/>
2. Open the file "step1_data_pre.py" in Spyder, then click "Run" to start the data preprocessing. Please note to check the dataset storage path. <br/>
3. After completing the data preprocessing, open the file "step2_Wytham_dataset_seg.py" in Spyder, then click "Run" to start tree segmentation. <br/>

Thanks to Henrich et al. for providing the dataset (https://doi.org/10.1016/j.ecoinf.2024.102888). <br/>
## Acknowledgments <br/>
This project wouldn't have been possible without the support and contributions of several individuals and resources. <br/>
Thanks to (in no particular order):
* https://github.com/eckerlab/TreeLearn
* https://github.com/ori-drs/aerial_terrestrial_registration
* https://github.com/ai4trees/pointtree
* https://github.com/anditockner/treeX

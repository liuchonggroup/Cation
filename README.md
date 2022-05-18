# Machine learning-guided identification of coordination polymer ligands for crystallizing separation of Cs/Sr
This repository is used to determine the Quantatitive Structure-Property Relationship (QSPR) from structural parameters of Metal-Ligand pairs (ML pairs) to their coordination bond properties. The workflow is as showed:
  
![image](https://user-images.githubusercontent.com/96228040/169002531-a70c4437-ea5c-41d7-a667-f2e0719ed7fe.png) 
  
When the repository has been cloned, users can run the scripts in the "main code" folder, directly. The scripts are:
1) mine_crystal: retrieving crystals' information containing specific metals from Cambridge Structural Database (CSD, https://www.ccdc.cam.ac.uk/).
2) data_process: extracting the structural parameters and the coordination bond properties of ML pairs from the retrieved crystals, where the structural parameters are abstructed as molecular graphs with nodes' features, i.e. the adjacency matrix and feature matrix.
3) training_pred: traning Graph Convolutional Network (GCN) to map the structural parameters to the coordination bond properties.

It should be note that the CSD and CSD Python API (https://www.ccdc.cam.ac.uk/solutions/csd-core/components/csd-python-api/) are required to get license and to install for running the scripts 1 and 2.

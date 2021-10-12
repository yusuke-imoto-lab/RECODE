scRNA-seq data - 10X chromium HDF5(h5) file
========

We show an exmaple for scRNA-seq data produced by 10X Chromium. 
We use a sample of `500 Human PBMCs, 3' LT v3.1, Chromium Controller` in `10X Genomics Datasets <https://www.10xgenomics.com/jp/resources/datasets>`_.  
The test data is directly avairable from `Feature / cell matrix HDF5 (filtered)` in `here <https://www.10xgenomics.com/jp/resources/datasets/500-human-pbm-cs-3-lt-v-3-1-chromium-controller-3-1-low-6-1-0>` (need register).


Import  ``numpy``, ``scipy``, and ``h5py`` in addlition to ``screcode``. 
.. code-block:: python

	import screcode
	import numpy as np
	import scipy
	import h5py


Imput data. 
.. code-block:: python

	h5_file = 'data/500_PBMC_3p_LT_Chromium_Controller_filtered_feature_bc_matrix.h5'
	input_h5 = h5py.File(h5_file,'r')
	data = scipy.sparse.csc_matrix((input_h5['matrix']['data'],input_h5['matrix']['indices'],input_h5['matrix']['indptr']),shape=input_h5['matrix']['shape']).toarray().T
	gene_list = [x.decode('ascii', 'ignore') for x in input_h5['matrix']['features']['name']]
	cell_list = np.array([x.decode('ascii', 'ignore') for x in input_h5['matrix']['barcodes']],dtype=object)


Apply scRECODE. 
.. code-block:: python
	screc = screcode.scRECODE()
	data_scRECODE = screc.fit_transform(data)

Check applicability. 
.. code-block:: python
	screc.check_applicability(save=True,save_filename='figure/ATAC_PBMC_applicability')

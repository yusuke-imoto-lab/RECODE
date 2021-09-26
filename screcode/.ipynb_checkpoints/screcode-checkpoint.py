import numpy as np
import sklearn.decomposition
import scipy.sparse
import h5py

def scRECODE(data, return_param=False):
    data_xRECODE = np.array(data, dtype=float)
    idx_gene = np.sum(data, axis=0) > 0
    data_sc = (data.T/np.sum(data, axis=1)).T[:, idx_gene]
    noise_var = np.mean(data.T/np.sum(data, axis=1) /
                        np.sum(data, axis=1), axis=1)[idx_gene]
    data_norm = (data_sc-np.mean(data_sc, axis=0))/np.sqrt(noise_var)
    pca = sklearn.decomposition.PCA()
    pca.fit_transform(data_norm)
    eigval_mod = np.append([pca.explained_variance_[i]-np.mean(pca.explained_variance_[i:])
                           for i in range(len(pca.explained_variance_)-1)], 0)
    ell_func = np.array([np.sum(pca.explained_variance_[i:]) for i in range(len(
        pca.explained_variance_))]) - (sum(idx_gene)-np.arange(len(pca.explained_variance_)))
    ell = np.sum(ell_func > 0)+1
    data_norm_RECODE = np.dot(np.dot(np.dot(data_norm, pca.components_[:ell].T), np.diag(
        np.sqrt(eigval_mod[:ell]/pca.explained_variance_[:ell]))), pca.components_[:ell])
    data_sc_xRECODE = data_norm_RECODE * \
        np.sqrt(noise_var)+np.mean(data_sc, axis=0)
    data_xRECODE[:, idx_gene] = (data_sc_xRECODE.T*np.sum(data, axis=1)).T
    data_xRECODE = np.where(data_xRECODE > 0, data_xRECODE, 0)
    if return_param:
        return data_xRECODE, ell
    else:
        return data_xRECODE


def scRECODE_h5(h5_file, decimals=5):
    input_h5 = h5py.File(h5_file, 'r')
    data = scipy.sparse.csc_matrix((input_h5['matrix']['data'], input_h5['matrix']['indices'],
                                   input_h5['matrix']['indptr']), shape=input_h5['matrix']['shape']).toarray().T
    gene_list = [x.decode('ascii', 'ignore')
                 for x in input_h5['matrix']['features']['name']]
    cell_list = np.array([x.decode('ascii', 'ignore')
                         for x in input_h5['matrix']['barcodes']], dtype=object)
    data_xRECODE, ell = scRECODE(data, return_param=True)
    data_xRECODE_csc = scipy.sparse.csc_matrix(
        np.round(data_xRECODE, decimals=decimals).T)
    outpur_h5 = '%s_xRECODE.h5' % (h5_file[:-3])
    with h5py.File(outpur_h5, 'w') as f:
        f.create_group('matrix')
        for key in input_h5['matrix'].keys():
            if key == 'data':
                f['matrix'].create_dataset(key, data=data_xRECODE_csc.data)
            elif key == 'indices':
                f['matrix'].create_dataset(key, data=data_xRECODE_csc.indices)
            elif key == 'indptr':
                f['matrix'].create_dataset(key, data=data_xRECODE_csc.indptr)
            elif key == 'shape':
                f['matrix'].create_dataset(key, data=data_xRECODE_csc.shape)
            elif type(input_h5['matrix'][key]) == h5py._hl.dataset.Dataset:
                f['matrix'].create_dataset(key, data=input_h5['matrix'][key])
            else:
                f['matrix'].create_group(key)
                for key_sub in input_h5['matrix'][key].keys():
                    f['matrix'][key].create_dataset(
                        key_sub, data=input_h5['matrix'][key][key_sub])
        f['matrix'].create_dataset('xRECODE_ell', data=ell)

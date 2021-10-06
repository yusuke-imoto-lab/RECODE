impoart numpy as np
import sklearn.decomposition
import scipy.sparse
import h5py


def RECODE(data,ell):
    pca = sklearn.decomposition.PCA()
    pca.fit_transform(data)
    eigval_mod = np.append([pca.explained_variance_[i]-np.mean(pca.explained_variance_[i:]) for i in range(len(pca.explained_variance_)-1)],0)
    data_RECODE = np.dot(np.dot(np.dot(data,pca.components_[:ell].T),np.diag(np.sqrt(eigval_mod[:ell]/pca.explained_variance_[:ell]))),pca.components_[:ell])
    return data_RECODE


class scRECODE():
    def __init__(
        self,
        return_param = False,
        acceleration = True,
        ell_max = 1000
    ):
        self.return_param = return_param
        self.acceleration = acceleration
        self.ell_max = ell_max
    
    def noise_variance_stabilizing_normalization(
        self,
        data,
        return_param=False
    ):
        ## probability data
        data_scaled = (data.T/self.data_nUMI).T
        data_scaled_mean = np.mean(data_scaled,axis=0)
        ## normalization
        noise_var = np.mean(data.T/self.data_nUMI/self.data_nUMI,axis=1)
        noise_var[noise_var==0] = 1
        data_norm = (data_scaled-np.mean(data_scaled,axis=0))/np.sqrt(noise_var)
        self.data_scaled = data_scaled
        self.data_scaled_mean = data_scaled_mean
        self.noise_var = noise_var
        if return_param == True:
            data_norm_var = np.var(data_norm,axis=0)
            n_sig = sum(data_norm_var>1)
            n_silent = sum(np.sum(data,axis=0)==0)
            n_nonsig = data.shape[1] - n_sig - n_silent
            param = {
                '#significant genes':n_sig,
                '#non-significant genes':n_nonsig,
                '#silent genes':n_silent
            }
            return data_norm, param
        else:
            return data_norm
    
    def fit_transform(
        self,
        data,
    ):
        print(self.ell_max)
        data_scRECODE = np.zeros(data.shape,dtype=float)
        idx_gene = np.sum(data,axis=0) > 0
        data_sc =(data.T/np.sum(data,axis=1)).T[:,idx_gene]
        noise_var = np.mean(data.T/np.sum(data,axis=1)/np.sum(data,axis=1),axis=1)[idx_gene]
        data_norm = (data_sc-np.mean(data_sc,axis=0))/np.sqrt(noise_var)
        pca = sklearn.decomposition.PCA()
        pca.fit_transform(data_norm)
        eigval_mod = np.append([pca.explained_variance_[i]-np.mean(pca.explained_variance_[i:]) for i in range(len(pca.explained_variance_)-1)],0)
        ell_func = np.array([np.sum(pca.explained_variance_[i:]) for i in range(len(pca.explained_variance_))]) - (sum(idx_gene)-np.arange(len(pca.explained_variance_)))
        ell = np.sum(ell_func>0)+1
        data_norm_RECODE = RECODE(data_norm,ell)
        data_sc_scRECODE = data_norm_RECODE*np.sqrt(noise_var)+np.mean(data_sc,axis=0)
        data_scRECODE[:,idx_gene] = (data_sc_scRECODE.T*np.sum(data,axis=1)).T
        data_scRECODE = np.where(data_scRECODE>0,data_scRECODE,0)
        if self.return_param:
            return data_scRECODE,ell
        else:
            return data_scRECODE


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

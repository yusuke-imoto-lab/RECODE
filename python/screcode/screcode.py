import anndata
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import pandas as pd
import sklearn.decomposition
import scipy.sparse
import scanpy
import seaborn as sns
import logging
import pandas as pd


class RECODE:
    def __init__(
        self,
        fast_algorithm=True,
        fast_algorithm_ell_ub=1000,
        seq_target="RNA",
        version=2,
        solver="auto",
        downsampling_rate=0.2,
        decimals=5,
        RECODE_key="RECODE",
        anndata_key="layers",
        random_state=0,
        log_normalize=True,
        target_sum=1e5,
        verbose=True,
    ):
        """
        RECODE (Resolution of curse of dimensionality in single-cell data analysis). A noise reduction method for single-cell sequencing data.

        Parameters
        ----------
        fast_algorithm : boolean, default=True
                If True, the fast algorithm is conducted. The upper bound of parameter ell is set by ``fast_algorithm_ell_ub``.

        fast_algorithm_ell_ub : int, default=1000
                Upper bound of parameter ell for the fast algorithm. Must be of range [1,infinity).

        seq_target : {'RNA','ATAC','Hi-C','Multiome'}, default='RNA'
                Sequencing target. If 'ATAC', the preprocessing (odd-even stabilization) will be performed before the regular algorithm.

        version : int default='2'
                Version of RECODE. Version 1 is the original algorithm (`Imoto-Nakamura et al. 2022 <https://doi.org/10.26508/lsa.202201591>`_.) Version 2 includes the eigenvector modification (`Imoto 2024 <https://doi.org/10.1101/2024.04.18.590054>`_.)
        
        solver : {'auto', 'full', 'randomized'}, default="auto"
                auto: set ``solver='randomized'`` if the number of samples (cells) are larger than 20,000. Otherwise set ``solver='full'``. 
                full: run learning process using the full input matrix. 
                randomized: run learning process involving computing SVD and estimating the essential dimension using downsampled data with the rate ``downsampling_rate``. 
        
        downsampling_rate : float, default=1000
                Downsampling rate, which is only relevant when ``solver='randomized'``. 

        decimals : int default='5'
                Number of decimals for round processed matrices.
        
        RECODE_key : string, default='RECODE'
                Key name of anndata to store the output. 
        
        anndata_key : {'layers','obsm'}, default='layers'
                Attribute of anndata where the output stored. 
        
        random_state : int or None, default=0
                Used when the 'randomized' solver is used.
        
        verbose : boolean, default=True
                If False, all running messages are not displayed.

        Attributes
        ----------
        cv_ : ndarray of shape (n_features,)
                Coefficient of variation of features (genes/peaks).

        log_ : dict
                Running log.

        noise_variance_ : ndarray of shape (n_features,)
                Noise variances of features (genes/peaks).

        normalized_variance_ : ndarray of shape (n_features,)
                Normalized variances of features (genes/peaks).

        significance_ : ndarray of shape (n_features,)
                Significance (significant/non-significant/silent) of features (genes/peaks).
        """
        self.fast_algorithm = fast_algorithm
        self.fast_algorithm_ell_ub = fast_algorithm_ell_ub
        self.seq_target = seq_target
        self.version = version
        self.solver = solver
        self.downsampling_rate = downsampling_rate
        self.decimals = decimals
        self.RECODE_key = RECODE_key
        self.anndata_key = anndata_key
        self.random_state = random_state
        self.log_normalize = log_normalize
        self.target_sum = target_sum
        self.verbose = verbose

        # Set unit and Unit based on seq_target
        if seq_target == "ATAC":
            self.unit, self.Unit = "peak", "Peak"
        elif seq_target == "Hi-C":
            self.unit, self.Unit = "contact", "Contact"
        elif seq_target == "Multiome":
            self.unit, self.Unit = "feature", "Feature"
        else:
            self.unit, self.Unit = "gene", "Gene"

        self.log_ = {"seq_target": self.seq_target}
        self.fit_idx = False

        self.logger = logging.getLogger("argument checking")
        if self.verbose:
            self.logger.setLevel(logging.WARNING)
        else:
            self.logger.setLevel(logging.ERROR)
            
    def _check_datatype(self, X):
        # if type(X) == anndata._core.anndata.AnnData:
        if isinstance(X, anndata.AnnData):
            adata = X
            if "feature_types" in adata.var.keys():
                if (set(adata.var["feature_types"]) == {"Gene Expression"}) & (
                    self.seq_target != "RNA"
                ):
                    self.logger.warning(
                        "Warning: Input data may be scRNA-seq data. Please add option seq_target='RNA' like screcode.RECODE(seq_target='RNA'). "
                    )
                elif (set(adata.var["feature_types"]) == {"Peaks"}) & (
                    self.seq_target != "ATAC"
                ):
                    self.logger.warning(
                        "Warning: Input data may be scATAC-seq data. Please add option seq_target='ATAC' like screcode.RECODE(seq_target='ATAC'). "
                    )
                elif (
                    set(adata.var["feature_types"]) == {"Gene Expression", "Peaks"}
                ) & (self.seq_target != "Multiome"):
                    self.logger.warning(
                        "Warning: Input data may be multiome (scRNA-seq + scATAC-seq) data. Please add option seq_target='Multiome' like screcode.RECODE(seq_target='Multiome'). "
                    )

            if scipy.sparse.issparse(adata.X):
                return adata.X.toarray()
            elif isinstance(adata.X, np.ndarray):
                return adata.X
            elif isinstance(adata.X, anndata._core.views.ArrayView):
                return np.array(adata.X)
            else:
                raise TypeError("Data type error: ndarray or anndata is available.")
        elif self.seq_target == "Multiome":
            raise TypeError(
                "Data type error: only anndata type is acceptable for multiome (scRNA-seq + scATAC-seq) data."
            )
        elif scipy.sparse.issparse(X):
            self.logger.warning(
                "RECODE does not support sparse input. The input and output are transformed as regular matricies. "
            )
            return X.toarray()
        # elif type(X) == np.ndarray:
        if isinstance(X, np.ndarray):
            return X
        else:
            raise TypeError("Data type error: ndarray or anndata is available.")

    def _total_scaling(self, X):
        X_total = np.sum(X, axis=1)
        X_total[X_total==0] = 1
        return X / X_total[:,np.newaxis]
    
    def _logp1(
            self,
            X,
            base=None
        ):
        return np.log(X+1) if base is None else np.log(X+1)/np.log(base)

    def _noise_variance_stabilizing_normalization(self, X):
        """
        Apply the noise-variance-stabilizing normalization to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
                Data matrix
        """
        d = X.shape[1]
        if d == self.d_train:
            X_ = X[:, self.idx_nonsilent]
        elif d == self.d_nonsilent:
            X_ = X
        else:
            raise TypeError("Dimension of data is not correct.")

        ## scaled X
        X_total = np.sum(X_, axis=1)
        X_total[X_total==0] = 1
        X_scaled = X_ / X_total[:,np.newaxis]
        X_norm = (X_scaled - self.X_scaled_mean) / np.sqrt(self.noise_var)
        self.X_total = X_total

        if d == self.d_train:
            X_norm_ = np.zeros(X.shape, dtype=float)
            X_norm_[:, self.idx_nonsilent] = X_norm
            return X_norm_
        elif d == self.d_nonsilent:
            return X_norm
            

    def _inv_noise_variance_stabilizing_normalization(self, X):
        """
        Apply the inverce transformation of noise-variance-stabilizing normalization to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
                Data matrix
        """
        X_norm_inv_temp = X * np.sqrt(self.noise_var) + self.X_scaled_mean
        X_norm_inv = X_norm_inv_temp * self.X_total[:,np.newaxis]
        return X_norm_inv

    def _ATAC_preprocessing(self, X):
        """
        Preprocessing of original ATAC-seq data (odd-even stabilization data).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
                Data matrix (scATAC-seq data)

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features)
                Preprecessed data matrix.
        """
        X_new = (X + 1) // 2
        return X_new

    def fit(self, X):
        """
        Fit the model to X. (Determine the transformation.)

        Parameters
        ----------
        X : ndarray or anndata of shape (n_samples, n_features).
                single-cell sequencing data matrix (row:cell, culumn:gene/peak).

        """
        X_mat = self._check_datatype(X)

        idx_act_cells = np.sum(X_mat,axis=1) > 0

        X_mat = X_mat[idx_act_cells]

        if self.solver == "auto":
            self.solver = "full" if X_mat.shape[0] < 10000 else "randomized"

        if self.solver ==  "randomized":
            if X_mat.shape[0] < 10000:
                self.logger.warning(
                    "Warning: randomized algorithm is for data with a large number of cells (>20000). \n"
                    "solver=\"full\" is recommended to keep the accuracy."
                )
            np.random.seed(self.random_state)
            cell_stat = np.random.choice(
                X_mat.shape[0], int(self.downsampling_rate * X.shape[0]), replace=False
            )
            X_mat = X_mat[cell_stat]
        else:
            if X.shape[0] > 20000:
                self.logger.warning(
                    "Warning: Regular RECODE uses high computational resources for data with a large number of cells. \n"
                    'solver=\"randomized\" is recommended. '
                )

        if np.linalg.norm(X_mat - np.array(X_mat, dtype=int)) > 0:
            self.logger.warning(
                "Warning: RECODE is applicable for count data (integer matrix). Plese make sure the data type."
            )
        self.idx_nonsilent = np.sum(X_mat, axis=0) > 0
        self.X_temp = X_mat[:, self.idx_nonsilent]
        if self.seq_target == "ATAC":
            self.X_temp = self._ATAC_preprocessing(self.X_temp)
        if self.seq_target == "Multiome":
            self.idx_atac = X.var["feature_types"][self.idx_nonsilent] == "Peaks"
            self.X_temp[:, self.idx_atac] = self._ATAC_preprocessing(
                self.X_temp[:, self.idx_atac]
            )
        X_nUMI = np.sum(self.X_temp, axis=1)
        X_scaled = self.X_temp / X_nUMI[:,np.newaxis]
        X_scaled_mean = np.mean(X_scaled, axis=0)
        noise_var = np.mean(
            self.X_temp / np.sum(self.X_temp, axis=1)[:,np.newaxis] / np.sum(self.X_temp, axis=1)[:,np.newaxis],
            axis=0,
        )
        noise_var[noise_var == 0] = 1
        X_norm = (X_scaled - X_scaled_mean) / np.sqrt(noise_var)
        X_norm_var = np.var(X_norm, axis=0)
        recode_ = RECODE_core(
            variance_estimate=False,
            fast_algorithm=self.fast_algorithm,
            fast_algorithm_ell_ub=self.fast_algorithm_ell_ub,
            version=self.version,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        recode_.fit(X_norm)

        # self.n_all = X.shape[0]
        # self.d_all = X.shape[1]
        self.n_train = X_mat.shape[0]
        self.d_train = X_mat.shape[1]
        self.d_nonsilent = sum(self.idx_nonsilent)
        self.noise_var = noise_var
        self.recode_ = recode_
        self.X_norm_var = X_norm_var
        self.X_fit_nUMI = X_nUMI
        self.X_scaled_mean = X_scaled_mean
        self.idx_sig = self.X_norm_var > 1
        self.idx_nonsig = self.idx_sig == False
        self.log_["#significant %ss" % self.unit] = int(sum(self.idx_sig))
        self.log_["#non-significant %ss" % self.unit] = int(sum(self.idx_nonsig))
        self.log_["#silent %ss" % self.unit] = int(X.shape[1] - sum(self.idx_sig) - sum(self.idx_nonsig))
        self.fit_idx = True

    def transform(self, X):
        """
        Transform X into RECODE-denoised data.

        Parameters
        ----------
        X : ndarray or anndata of shape (n_samples, n_features)
                Single-cell sequencing data matrix (row:cell, culumn:gene/peak).

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features)
                RECODE-denoised data matrix.
        """
        X_mat = self._check_datatype(X)
        if self.fit_idx == False:
            raise TypeError("Run fit before transform.")
        if X_mat.shape[1] != self.d_train:
            raise TypeError(
                "RECODE requires the same dimension as that of fitted data."
            )
        idx_act_cells = np.sum(X_mat,axis=1) > 0

        X_ = X_mat[np.ix_(idx_act_cells, self.idx_nonsilent)]
        X_norm = self._noise_variance_stabilizing_normalization(X_)
        X_norm_RECODE_, X_ess, _, _ = self.recode_.transform(X_norm,return_ess=True)
        X_norm_RECODE = np.zeros(X_mat.shape, dtype=float)
        X_norm_RECODE[np.ix_(idx_act_cells, self.idx_nonsilent)] = X_norm_RECODE_
        X_RECODE = np.zeros(X_mat.shape, dtype=float)
        X_RECODE[np.ix_(idx_act_cells, self.idx_nonsilent)] = self._inv_noise_variance_stabilizing_normalization(X_norm_RECODE_)
        X_RECODE = np.where(X_RECODE > 0, X_RECODE, 0)
        X_RECODE = np.round(X_RECODE, decimals=self.decimals)
        X_norm_RECODE = np.round(X_norm_RECODE, decimals=self.decimals)
        self.log_["#silent %ss" % self.unit] = int(sum(np.sum(X_mat, axis=0) == 0))
        self.log_["ell"] = int(self.recode_.ell)
        if self.recode_.ell == self.fast_algorithm_ell_ub:
            self.logger.warning(
                "Acceleration error: the value of ell may not be optimal. Set 'fast_algorithm=False' or larger fast_algorithm_ell_ub.\n"
                "Ex. X_new = screcode.RECODE(fast_algorithm=False).fit_transform(X)"
            )
        self.X_trans = np.round(X_mat, decimals=self.decimals)
        self.X_RECODE = X_RECODE
        self.noise_variance_ = np.zeros(X_mat.shape[1], dtype=float)
        self.noise_variance_[self.idx_nonsilent] = self.noise_var
        self.normalized_variance_ = np.zeros(X_mat.shape[1], dtype=float)
        self.normalized_variance_[self.idx_nonsilent] = self.X_norm_var

        X_RECODE_ss = np.median(np.sum(X_RECODE[np.ix_(idx_act_cells, self.idx_nonsilent)], axis=1))*self._total_scaling(X_RECODE[np.ix_(idx_act_cells, self.idx_nonsilent)])
        self.cv_ = np.zeros(X.shape[1], dtype=float)
        self.cv_[self.idx_nonsilent] = np.std(X_RECODE_ss, axis=0) / np.mean(X_RECODE_ss, axis=0)

        self.significance_ = np.empty(X.shape[1], dtype=object)
        self.significance_[self.normalized_variance_ == 0] = "silent"
        self.significance_[self.normalized_variance_ > 0] = "non-significant"
        self.significance_[self.normalized_variance_ > 1] = "significant"
        self.num_invalid_cells = sum(idx_act_cells==False)
        self.n_trans = X.shape[0]
        self.d_trans = X.shape[1]

        if type(X) == anndata._core.anndata.AnnData:
            X_out = anndata.AnnData.copy(X)
            if self.anndata_key == "obsm":
                X_out.obsm[self.RECODE_key] = X_RECODE
                X_out.obsm[f"{self.RECODE_key}_NVSN"] = X_norm_RECODE
                if self.verbose:
                    print(f"Normalized data are stored as \"{self.RECODE_key}\" in adata.obsm")
            else:
                X_out.layers[self.RECODE_key] = X_RECODE
                X_out.layers[f"{self.RECODE_key}_NVSN"] = X_norm_RECODE
                if self.verbose:
                    print(f"Normalized data are stored as \"{self.RECODE_key}\" in adata.layers")
            X_out.uns[f"{self.RECODE_key}_essential"] = X_ess
            X_out.var[f"{self.RECODE_key}_noise_variance"] = self.noise_variance_
            X_out.var[f"{self.RECODE_key}_NVSN_variance"] = self.normalized_variance_
            X_out.var[f"{self.RECODE_key}_significance"] = self.significance_
            if self.log_normalize==True:
                self.lognormalize(X_out, target_sum=self.target_sum)
                X_out.var[f"{self.RECODE_key}_means"] = X_out.layers[f"{self.RECODE_key}_log"].mean(axis=0)
        else:
            X_out = X_RECODE

        return X_out

    def fit_transform(self, X):
        """
        Fit the model with X and transform X into RECODE-denoised data.

        Parameters
        ----------
        X : ndarray/anndata of shape (n_samples, n_features)
                Tranceforming single-cell sequencing data matrix (row:cell, culumn:gene/peak).

        Returns
        -------
        X_new : ndarray/anndata (the same format as input)
                Denoised data matrix.
        """
        start_time = datetime.datetime.now()
        if self.verbose:
            if self.seq_target in ["RNA", "ATAC", "Hi-C"]:
                print("start RECODE for sc%s-seq data" % self.seq_target)
            if self.seq_target in ["Multiome"]:
                print("start RECODE for %s data" % self.seq_target)

        self.fit(X)
        X_RECODE = self.transform(X)
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int(elapsed_time.microseconds / 1000)
        self.elapsed_time = f"{hours}h {minutes}m {seconds}s"
        self.log_[
            "Elapsed time"
        ] = f"{hours}h {minutes}m {seconds}s {milliseconds:03}ms"
        self.log_["solver"] = self.solver
        if self.log_["solver"] == "randomized":
            self.log_["#train_data"] = self.n_train
        if self.verbose:
            print("end RECODE for sc%s-seq" % self.seq_target)
            print("log:", self.log_)
        return X_RECODE

    def transform_integration(
        self,
        X,
        meta_data=None,
        batch_key="batch",
        integration_method = "harmony",
        integration_method_params = {},
    ):
        """
        Transform X into RECODE-denoised data.

        Parameters
        ----------
        X : ndarray or anndata of shape (n_samples, n_features)
                Single-cell sequencing data matrix (row:cell, culumn:gene/peak).

        meta_data : ndarray (n_samples, 1) or DataFrame (n_samples, *)

        batch_key : string or list, default='batch'
                Key name(s) in ``meta_data`` denoting batch. 
        
        integration_method : {'harmony','mnn','scanorama','scvi'}, default='harmony'
                A batch correction method used in iRECODE. 

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features)
                RECODE-denoised data matrix.
        """
        X_mat = self._check_datatype(X)
        if self.fit_idx == False:
            raise TypeError("Run fit before transform.")
        if X_mat.shape[1] != self.d_train:
            raise TypeError(
                "RECODE requires the same dimension as that of training (fit) data."
            )
        if X_mat.shape[1] == len(self.idx_nonsilent):
            X_ = X_mat[:, self.idx_nonsilent]
        else:
            X_ = X_mat
        X_norm = self._noise_variance_stabilizing_normalization(X_)
        _, X_ess, U_ell, Xmean = self.recode_.transform(X_norm,return_ess=True)
        if isinstance(batch_key, str):
            batch_key = [batch_key]
        X_norm_RECODE = self.recode_.transform(X_norm,return_ess=True)
        if type(X) == anndata._core.anndata.AnnData:
            batch_key_ = batch_key.copy()
            batch_key = [b_ for b_ in batch_key if b_ in X.obs.keys()]
            if len(batch_key) == 0:
                raise ValueError(
                    "No batch key \"%s\" in adata.obs. Add batch key or specify a \"batch_key\"" % batch_key_
                )
            else:
                meta_data_ = {batch_key[i]: np.array(X.obs[batch_key[i]], dtype="object") for i in range(len(batch_key))}
        elif type(meta_data) == np.ndarray:
            if len(meta_data.shape) == len(batch_key):
                meta_data_ = {batch_key[i]: np.array(meta_data[i], dtype="object") for i in range(len(batch_key))}
            else:
                raise ValueError("meta_data (np.ndarray) should be the same dimension as the batch_key")
        elif (type(meta_data) == anndata._core.views.DataFrameView) | (type(meta_data) == pd.core.frame.DataFrame):
            for b_ in batch_key:
                if b_ not in meta_data.keys():
                    raise ValueError(
                        "No batch key \"%s\" in meta_data. Add batch key or specify a \"batch_key\"" % b_
                    )
            meta_data_ = {b_:np.array(meta_data[b_].values,dtype="object") for b_ in batch_key}
        else:
            raise TypeError(
                    "No batch data. Add batch indices in \"meta_data\""
                    )
        adata_ = anndata.AnnData(
            X_ess,
            obs = meta_data_,
            obsm = {"X":X_ess},
            # dtype=X_ess.dtype,
        )
        if integration_method == "harmony":
            scanpy.external.pp.harmony_integrate(adata_, basis='X',adjusted_basis='X_integrated',key=batch_key,verbose=False,**integration_method_params)
            X_ess_merge = adata_.obsm["X_integrated"]
        elif integration_method == "bbknn":
            scanpy.external.pp.bbknn(adata_, batch_key=batch_key, use_rep='X',**integration_method_params)
            X_ess_merge = adata_.X
        elif integration_method == "scanorama":
            scanpy.external.pp.scanorama_integrate(adata_, key=batch_key, basis='X',adjusted_basis='X_integrated',verbose=False,**integration_method_params)
            X_ess_merge = adata_.obsm["X_integrated"]
        elif integration_method == "mnn":
            batches = [' '.join([adata_.obs[b_][i] for b_ in batch_key]) for i in range(adata_.shape[0])]
            data_ = [adata_.X[batches==b_] for b_ in np.unique(batches)]
            mnn_out = scanpy.external.pp.mnn_correct(*data_, var_index=np.arange(adata_.shape[1]),verbose=False,cos_norm_out=False,**integration_method_params)
            X_ess_merge = mnn_out[0]
        elif integration_method == "scvi":
            try:
                import scvi
            except ImportError:
                raise ImportError("\nplease install scvi:\n\n\tpip install scvi-tools")
            adata_.X = adata_.X-np.min(adata_.X)
            scvi.model.SCVI.setup_anndata(adata_, batch_key="batch")
            model = scvi.model.SCVI(adata_,gene_likelihood="normal",n_latent=adata_.shape[1],dropout_rate=0,dispersion='gene-batch',**integration_method_params)
            model.train()
            X_ess_merge = model.get_latent_representation() + np.min(adata_.X)
        elif integration_method == "cca":
            from sklearn.cross_decomposition import CCA
            X_ess_merge = adata_.X
            for b_ in batch_key:
                batch_set_,counts_ = np.unique(adata_.obs[b_],return_counts=True)
                idx_batch = np.argsort(counts_)#[::-1]
                X_merged = X_ess_merge[adata_.obs[b_] == batch_set_[idx_batch[0]]]
                for i in range(len(idx_batch)-1):
                    Y_ = X_ess_merge[adata_.obs[b_] == batch_set_[idx_batch[i+1]]]
                    indices = np.random.choice(X_merged.shape[0], size=Y_.shape[0], replace=False)
                    X_ = X_merged[indices]
                    cca = CCA(n_components=adata_.shape[1])
                    cca.fit(X_, Y_)
                    X_c, Y_c = cca.transform(X_merged, Y_)
                    X_merged = np.concatenate([X_c, Y_c])
                X_ess_merge = X_merged
        else:
            raise ValueError("No integration method \"%s\". Choice from %s" % integration_method,["harmony","bbknn","scanorama","mnn"])
        X_norm_RECODE_merge_ = np.dot(X_ess_merge, U_ell) + Xmean
        X_norm_RECODE_merge = np.zeros(X_mat.shape, dtype=float)
        X_norm_RECODE_merge[:, self.idx_nonsilent] = X_norm_RECODE_merge_
        X_RECODE = np.zeros(X_mat.shape, dtype=float)
        X_RECODE[:, self.idx_nonsilent] = self._inv_noise_variance_stabilizing_normalization(X_norm_RECODE_merge_)
        X_RECODE = np.where(X_RECODE > 0, X_RECODE, 0)
        self.log_["#silent %ss" % self.unit] = int(sum(np.sum(X_mat, axis=0) == 0))
        self.log_["ell"] = self.recode_.ell
        if self.recode_.ell == self.fast_algorithm_ell_ub:
            self.logger.warning(
                "Acceleration error: the value of ell may not be optimal. Set 'fast_algorithm=False' or larger fast_algorithm_ell_u@b.\n"
                "Ex. X_new = screcode.RECODE(fast_algorithm=False).fit_transform(X)"
            )
        self.X_trans = np.round(X_mat, decimals=self.decimals)
        self.X_RECODE = np.round(X_RECODE, decimals=self.decimals)
        self.noise_variance_ = np.zeros(X_mat.shape[1], dtype=float)
        self.noise_variance_[self.idx_nonsilent] = self.noise_var
        self.normalized_variance_ = np.zeros(X_mat.shape[1], dtype=float)
        self.normalized_variance_[self.idx_nonsilent] = self.X_norm_var

        # X_RECODE_ss = (
        #     np.median(np.sum(X_RECODE[:, self.idx_nonsilent], axis=1))
        #     * X_RECODE[:, self.idx_nonsilent].T
        #     / np.sum(X_RECODE[:, self.idx_nonsilent], axis=1)
        # ).T
        X_RECODE_ss = (
            np.median(np.sum(X_RECODE[:, self.idx_nonsilent], axis=1))
            * X_RECODE[:, self.idx_nonsilent]
            / np.sum(X_RECODE[:, self.idx_nonsilent], axis=1)[:,np.newaxis]
        )
        self.cv_ = np.zeros(X.shape[1], dtype=float)
        self.cv_[self.idx_nonsilent] = np.std(X_RECODE_ss, axis=0) / np.mean(
            X_RECODE_ss, axis=0
        )

        self.significance_ = np.empty(X.shape[1], dtype=object)
        self.significance_[self.normalized_variance_ == 0] = "silent"
        self.significance_[self.normalized_variance_ > 0] = "non-significant"
        self.significance_[self.normalized_variance_ > 1] = "significant"

        if type(X) == anndata._core.anndata.AnnData:
            X_out = anndata.AnnData.copy(X)
            if self.anndata_key == "obsm":
                X_out.obsm[self.RECODE_key] = X_RECODE
                X_out.obsm[f"{self.RECODE_key}_NVSN"] = X_norm_RECODE_merge
            else:
                X_out.layers[self.RECODE_key] = X_RECODE
                X_out.layers[f"{self.RECODE_key}_NVSN"] = X_norm_RECODE_merge
            X_out.uns[f"{self.RECODE_key}_essential"] = X_ess
            X_out.var[f"{self.RECODE_key}_noise_variance"] = self.noise_variance_
            X_out.var[f"{self.RECODE_key}_NVSN_variance"] = self.normalized_variance_
            X_out.var[f"{self.RECODE_key}_significance"] = self.significance_
        else:
            X_out = X_RECODE

        return X_out

    def fit_transform_integration(
        self,
        X,
        meta_data=None,
        batch_key="batch",
        integration_method = "harmony",
        integration_method_params = {},
    ):
        """
        Fit the model with X and transform X into RECODE-denoised data.

        Parameters
        ----------
        X : ndarray/anndata of shape (n_samples, n_features)
                Tranceforming single-cell sequencing data matrix (row:cell, culumn:gene/peak).
        integration_method : {'harmony','mnn','scanorama','scvi'}, default='harmony'
                A batch correction method used in iRECODE. 

        Returns
        -------
        X_new : ndarray/anndata (the same format as input)
                Denoised data matrix.
        """
        start_time = datetime.datetime.now()
        if self.verbose:
            if self.seq_target in ["RNA", "ATAC", "Hi-C"]:
                print("start RECODE integration for sc%s-seq data" % self.seq_target)
            if self.seq_target in ["Multiome"]:
                print("start RECODE integration for %s data" % self.seq_target)
        
        self.fit(X)
        X_RECODE = self.transform_integration(X, meta_data, batch_key, integration_method, integration_method_params)
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int(elapsed_time.microseconds / 1000)
        self.elapsed_time = f"{hours}h {minutes}m {seconds}s"
        self.log_["Elapsed time"] = f"{hours}h {minutes}m {seconds}s {milliseconds:03}ms"
        self.log_["solver"] = self.solver
        if self.log_["solver"] == "randomized":
            self.log_["#test_data"] = int(self.downsampling_rate * X.shape[0])
        if self.verbose:
            print("end RECODE for sc%s-seq" % self.seq_target)
            print("log:", self.log_)
        return X_RECODE
        
    
    def lognormalize(
            self,
            X,
            base=None,
            target_sum=1e4,
            key = None,
            
    ):
        """
        Standard normalization: Normalize counts per cell and then logarithmize it:
        :math:`x_{ij}^{\\rm log} = \\log(x_{ij}^{\\rm norm} + 1)`, where :math:`x_{ij}^{\\rm norm} = c*x_{ij}/\\sum_{i}x_{ij}` and :math:`x_{ij}' is the count calue of :math:`i'th cell and :math:`j'th gene.

        Parameters
        ----------
        X : ndarray or anndata of shape (n_samples, n_features).
                single-cell sequencing data matrix (row:cell, culumn:gene/peak).
        
        base : positive number or None, default=None
                Base of the logarithm. If None, natural logarithm is used.

        target_sum : float, default=1e4,
                Total value after count normalization, corresponding the coefficient :math:`c` above.
        
        RECODE_key : string, default=None
                Key name of anndata to store the output. If None, the RECODE_key that is set initially is used. 

        """
        if key == None:
            key = self.RECODE_key
        
        if type(X) == anndata._core.anndata.AnnData:
            if self.anndata_key == "obsm":
                X_mat_ = X.obsm[key]
            else:
                X_mat_ = X.layers[key]
        else:
            X_mat_ = self._check_datatype(X)
        
        X_ss = target_sum*self._total_scaling(X_mat_)
        X_log = self._logp1(X_ss,base)
        
        if type(X) == anndata._core.anndata.AnnData:
            if self.anndata_key == "obsm":
                X.obsm[f"{key}_norm"] = X_ss
                X.obsm[f"{key}_log"] = X_log
            else:
                X.layers[f"{key}_norm"] = X_ss
                X.layers[f"{key}_log"] = X_log
            X.var[f"{key}_denoised_variance"] = np.var(X_log, axis=0)
            if self.verbose:
                print("Normalized data are stored as \"%s\" and \"%s\" in adata.layers" % (f"{key}_norm",f"{key}_log"))
            return X
        else:
            return X_log
    
    def highly_variable_genes(
        self,
        adata,
        n_top_genes=2000,
        min_variance=None,
        max_variance=None,
        min_mean=None,
        max_mean=None,
        inplace=True,
        RECODE_key="RECODE",
        mean_key="means",
        variance_key="denoised_variance",
        output_key="highly_variable",
        verbose=True,
    ):
        """
        Select highly variable genes using normalized variance.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix.
        n_top_genes : int or None
            Number of top genes to select by normalized variance. If None, use thresholds.
        min_variance : float or None
            Minimum normalized variance for a gene to be considered. If None, do not filter by variance.
        max_variance : float or None
            Maximum normalized variance for a gene to be considered. If None, do not filter by variance.
        min_mean : float or None
            Minimum mean expression for a gene to be considered. If None, do not filter by mean.
        max_mean : float or None
            Maximum mean expression for a gene to be considered. If None, do not filter by mean.
        inplace : bool
            If True, adds a boolean mask to adata.var[output_key].
        mean_key : str
            Key in adata.var for mean values.
        variance_key : str
            Key in adata.var for normalized variance.
        output_key : str
            Key in adata.var to store the boolean mask.

        Returns
        -------
        None or pandas.DataFrame
            If inplace=False, returns a copy of adata.var with the mask added.
        """
        if RECODE_key not in adata.layers:
            raise ValueError(f"{RECODE_key} not found in adata.layers. Please conduct recode = screcode.RECODE and recode.fit_transform(adata) first.")
        elif f"{RECODE_key}_{variance_key}" not in adata.var:
            raise ValueError(f"\"{RECODE_key}_{variance_key}\" not found in adata.var. Please change key RECODE_key or variance_key. ")

        mean = adata.var[f"{RECODE_key}_{mean_key}"].values
        norm_var = adata.var[f"{RECODE_key}_{variance_key}"].values

        # mean filter (optional)
        if min_mean is not None or max_mean is not None:
            gene_filter = np.ones_like(mean, dtype=bool)
            if min_mean is not None:
                gene_filter &= (mean > min_mean)
            if max_mean is not None:
                gene_filter &= (mean < max_mean)
        else:
            gene_filter = np.ones_like(mean, dtype=bool)

        filtered_norm_var = norm_var[gene_filter]

        # variance filter (optional)
        if min_variance is not None or max_variance is not None:
            disp_filter = np.ones_like(filtered_norm_var, dtype=bool)
            if min_variance is not None:
                disp_filter &= (filtered_norm_var > min_variance)
            if max_variance is not None:
                disp_filter &= (filtered_norm_var < max_variance)
            selected = np.zeros(len(norm_var), dtype=bool)
            idx = np.where(gene_filter)[0][disp_filter]
            selected[idx] = True
        else:
            selected = gene_filter.copy()

        # n_top_genes
        if n_top_genes is not None:
            top_idx = np.argsort(norm_var)[::-1][:n_top_genes]
            selected = np.zeros(len(norm_var), dtype=bool)
            selected[top_idx] = True

        if inplace:
            adata.var[f"{RECODE_key}_{output_key}"] = selected
            if verbose or self.verbose:
                print(f"Highly variable genes are stored in adata.var['{RECODE_key}_{output_key}']")
        else:
            result = adata.var.copy()
            result[f"{RECODE_key}_{output_key}"] = selected
            return result

    def check_applicability(
        self,
        title="",
        figsize=(10, 5),
        ps=2,
        save=False,
        save_filename="check_applicability",
        save_format="png",
        dpi=None,
        show=True,
    ):
        """
        Check the applicability of RECODE.
        Before using this function, you have to conduct ``fit(X)`` or ``fit_transform(X)`` for the target data matrix ``X``.

        Parameters
        ----------
        title : str, default=''
                Figure title.

        figsize : 2-tuple of floats, default=(10,5)
                Figure dimension ``(width, height)`` in inches.

        ps : float, default=10,
                Point size.

        save : bool, default=False
                If True, save the figure.

        save_filename : str, default= 'check_applicability',
                File name (path) of save figure.

        save_format : {'png', 'pdf', 'svg'}, default= 'png',
                File format of save figure.

        dpi: float or None, default=None
                Dots per inch.
        """
        # X_scaled = (self.X_temp.T / np.sum(self.X_temp, axis=1)).T
        X_scaled = self.X_temp / np.sum(self.X_temp, axis=1)[:,np.newaxis]
        X_norm = self._noise_variance_stabilizing_normalization(self.X_temp)
        norm_var = np.var(X_norm, axis=0, ddof=1)
        x, y = np.mean(X_scaled, axis=0), norm_var
        idx_nonsig, idx_sig = y <= 1, y > 1
        fig = plt.figure(figsize=figsize)
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        spec = matplotlib.gridspec.GridSpec(
            ncols=2, nrows=1, width_ratios=[4, 1], wspace=0.0
        )
        ax0 = fig.add_subplot(spec[0])
        if self.seq_target == "Multiome":
            ax0.scatter(
                x[idx_sig & (self.idx_atac == False)],
                y[idx_sig & (self.idx_atac == False)],
                color="b",
                s=ps,
                label="significant genes",
                zorder=2,
                marker="o",
                facecolor="None",
            )
            ax0.scatter(
                x[idx_sig & self.idx_atac],
                y[idx_sig & self.idx_atac],
                color="lightblue",
                s=ps,
                label="significant peaks",
                zorder=2,
                marker="o",
                facecolor="None",
            )
            ax0.scatter(
                x[idx_nonsig & (self.idx_atac == False)],
                y[idx_nonsig & (self.idx_atac == False)],
                color="r",
                s=ps,
                label="non-significant genes",
                zorder=3,
                marker="x",
            )
            ax0.scatter(
                x[idx_nonsig & self.idx_atac],
                y[idx_nonsig & self.idx_atac],
                color="orange",
                s=ps,
                label="non-significant peaks",
                zorder=3,
                marker="x",
            )
        else:
            ax0.scatter(
                x[idx_sig],
                y[idx_sig],
                color="b",
                s=ps,
                label="significant %s" % self.unit,
                zorder=2,
            )
            ax0.scatter(
                x[idx_nonsig],
                y[idx_nonsig],
                color="r",
                s=ps,
                label="non-significant %s" % self.unit,
                zorder=3,
            )
        ax0.axhline(1, color="gray", ls="--", lw=2, zorder=1)
        ax0.set_xscale("log")
        ax0.set_yscale("log")
        ax0.set_title(title, fontsize=14)
        ax0.set_xlabel("Mean of scaled data", fontsize=14)
        ax0.set_ylabel("NVSN variance", fontsize=14)
        ax0.legend(
            loc="upper left", borderaxespad=0, fontsize=14, markerscale=5
        ).get_frame().set_alpha(0)
        ylim = ax0.set_ylim()
        ax1 = fig.add_subplot(spec[1])
        sns.kdeplot(y=np.log10(norm_var[norm_var > 0]), color="k", fill=True, ax=ax1)
        ax1.axhline(0, c="gray", ls="--", lw=2, zorder=1)
        ax1.axvline(0, c="k", ls="-", lw=1, zorder=1)
        ax1.set_ylim(np.log10(ax0.set_ylim()))
        ax1.tick_params(labelbottom=True, labelleft=False, bottom=True)
        ax1.set_xlabel("Density", fontsize=14)
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.tick_params(left=False)
        ax1.patch.set_alpha(0)
        #
        x = np.linspace(ax1.set_ylim()[0], ax1.set_ylim()[1], 1000)
        dens = scipy.stats.gaussian_kde(np.log10(norm_var[norm_var > 0]))(x)
        peak_val = x[np.argmax(dens)]
        rate_low_var = np.sum(norm_var[norm_var > 0] < 0.90) / len(
            norm_var[norm_var > 0]
        )
        applicability = "Unknown"
        backcolor = "w"
        if (rate_low_var < 0.01) and (np.abs(peak_val) < 0.1):
            applicability = "Class A (strongly applicable)"
            backcolor = "lightgreen"
        elif rate_low_var < 0.01:
            applicability = "Class B (weakly applicable)"
            backcolor = "yellow"
        else:
            applicability = "Class C (inapplicabile)"
            backcolor = "tomato"
        ax0.text(
            0.99,
            0.982,
            applicability,
            va="top",
            ha="right",
            transform=ax0.transAxes,
            fontsize=14,
            backgroundcolor=backcolor,
        )
        self.log_["Applicability"] = applicability
        self.log_["Rate of 0 < normalized variance < 0.9"] = "{:.0%}".format(
            rate_low_var
        )
        self.log_["Peak density of normalized variance"] = 10**peak_val
        if self.verbose:
            print("applicabity:", applicability)
        if save:
            plt.savefig(
                "%s.%s" % (save_filename, save_format), dpi=dpi, bbox_inches="tight"
            )
        if show:
            plt.show()

    def plot_procedures(
        self,
        titles=(
            "Original data",
            "Normalized data",
            "Projected data",
            "Variance-modified data",
            "Denoised data",
        ),
        figsize=(7, 5),
        save=False,
        save_filename="RECODE_procedures",
        save_filename_foots=(
            "1_Original",
            "2_Normalized",
            "3_Projected",
            "4_Variance-modified",
            "5_Denoised",
        ),
        save_format="png",
        dpi=None,
        show=True,
    ):
        """
        Plot procedures of RECODE. The vertical axes of feature are sorted by the mean.

        Parameters
        ----------
        titles : 5-tuple of str, default=('Original data','Normalized data','Projected data','Variance-modified data','Denoised data')
                Figure titles.

        figsize : 2-tuple of floats, default=(7,5)
                Figure dimension ``(width, height)`` in inches.

        save : bool, default=False
                If True, save the figure.

        save_filename : str, default='RECODE_procedures',
                File name (path) of save figure (head).

        save_filename_foots : 5-tuple of str, default=('1_Original','2_Normalized','3_Projected','4_Variance-modified','5_Denoised'),
                File name (path) of save figure (foot).

        save_format : {'png', 'pdf', 'svg'}, default= 'png',
                File format of save figure.

        dpi: float or None, default=None
                Dots per inch.
        """

        if self.seq_target == "ATAC":
            title = "ATAC preprocessing"
            foot = "0_ATAC_preprocessing"
            self.plot_ATAC_preprocessing(
                title=title,
                figsize=figsize,
                save=save,
                save_filename="%s_%s" % (save_filename, foot),
                save_format=save_format,
                dpi=dpi,
                show=show,
            )

        self.plot_original_data(
            title=titles[0],
            figsize=figsize,
            save=save,
            save_filename="%s_%s" % (save_filename, save_filename_foots[0]),
            save_format=save_format,
            dpi=dpi,
            show=show,
        )

        self.plot_normalized_data(
            title=titles[1],
            figsize=figsize,
            save=save,
            save_filename="%s_%s" % (save_filename, save_filename_foots[1]),
            save_format=save_format,
            dpi=dpi,
            show=show,
        )

        self.plot_projected_data(
            title=titles[2],
            figsize=figsize,
            save=save,
            save_filename="%s_%s" % (save_filename, save_filename_foots[2]),
            save_format=save_format,
            dpi=dpi,
            show=show,
        )

        self.plot_variance_modified_data(
            title=titles[3],
            figsize=figsize,
            save=save,
            save_filename="%s_%s" % (save_filename, save_filename_foots[3]),
            save_format=save_format,
            dpi=dpi,
            show=show,
        )

        self.plot_denoised_data(
            title=titles[4],
            figsize=figsize,
            save=save,
            save_filename="%s_%s" % (save_filename, save_filename_foots[4]),
            save_format=save_format,
            dpi=dpi,
            show=show,
        )

    def report(
        self,
        figsize=(8.27, 11.69),
        save=False,
        save_filename="report",
        save_format="png",
        dpi=None,
        show=True,
        base = None,
    ):
        """
        Check the applicability of RECODE.
        Before using this function, you have to conduct ``fit(X)`` or ``fit_transform(X)`` for the target data matrix ``X``.

        Parameters
        ----------
        title : str, default=''
                Figure title.

        figsize : 2-tuple of floats, default=(8.27,11.69)
                Figure dimension ``(width, height)`` in inches.

        save : bool, default=False
                If True, save the figure.

        save_filename : str, default= 'report',
                File name (path) of save figure.

        save_format : {'png', 'pdf', 'svg'}, default= 'png',
                File format of save figure.

        dpi: float or None, default=None
                Dots per inch.
        """
        fs_label = 14
        X_scaled = self._total_scaling(self.X_temp)
        X_norm = self._noise_variance_stabilizing_normalization(self.X_temp)
        norm_var = np.var(X_norm, axis=0, ddof=1)
        target_sum = self.target_sum
        X_ss_log = self._logp1(target_sum * self._total_scaling(self.X_trans[:, self.idx_nonsilent]),base)
        X_RECODE_ss_log = self._logp1(target_sum * self._total_scaling(self.X_RECODE[:, self.idx_nonsilent]),base)
        # X_ss_log = np.log(target_sum * self.X_trans[:, self.idx_nonsilent] / np.sum(self.X_trans, axis=1)[:,np.newaxis] + 1)
        # X_RECODE_ss_log = np.log(target_sum * self.X_RECODE[:, self.idx_nonsilent] / np.sum(self.X_RECODE, axis=1)[:,np.newaxis] + 1)
        plot_EV = self.recode_.PCA_Ev[self.recode_.PCA_Ev > 0]
        n_EV = len(plot_EV)
        plot_EV_mod = np.zeros(n_EV)
        plot_EV_mod[: self.recode_.ell] = self.recode_.PCA_Ev_NRM[: self.recode_.ell]
        #
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="w")
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.subplots_adjust(left=0.05, right=0.97, bottom=0.01, top=0.98)
        gs_master = GridSpec(nrows=200, ncols=100, wspace=0, hspace=0)
        gs = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[8, 0:50])
        ax = fig.add_subplot(gs[0, 0])
        ax.text(0, 0, "RECODE report", fontsize=25, fontweight="bold")
        ax.axis("off")
        gs = GridSpecFromSubplotSpec(
            nrows=1, ncols=1, subplot_spec=gs_master[12:28, 0:100]
        )
        ax = fig.add_subplot(gs[0, 0])
        ax.patch.set_facecolor("gainsboro")
        ax.tick_params(
            labelbottom=False, labelleft=False, labelright=False, labeltop=False
        )
        ax.tick_params(bottom=False, left=False, right=False, top=False)
        gs = GridSpecFromSubplotSpec(
            nrows=1, ncols=1, subplot_spec=gs_master[8, 50:100]
        )
        ax = fig.add_subplot(gs[0, 0])
        now = datetime.datetime.today()
        ax.text(
            1,
            0,
            "Date: %d-%02d-%02d %02d:%02d:%02d"
            % (now.year, now.month, now.day, now.hour, now.minute, now.second),
            fontsize=12,
            ha="right",
        )
        ax.axis("off")
        gs = GridSpecFromSubplotSpec(
            nrows=1, ncols=1, subplot_spec=gs_master[16:26, 2:30]
        )
        ax = fig.add_subplot(gs[0, 0])
        ax.text(0, 1, "Method: %s" % self.log_["seq_target"], fontsize=12)
        ax.text(0, 0.5, "nCells: %s" % self.n_trans, fontsize=12)
        ax.text(0, 0.0, "n%ss: %s" % (self.Unit, self.d_train), fontsize=12)
        # ax.text(0,0.0,'Method: %s' % self.log_['seq_target'],fontsize=12)
        ax.axis("off")
        gs = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[16:26, 25:64])
        ax = fig.add_subplot(gs[0, 0])
        ax.text(
            0,
            1,
            "#significant %ss: %s"
            % (self.unit, self.log_["#significant %ss" % self.unit]),
            fontsize=12,
        )
        ax.text(
            0,
            0.5,
            "#non-significant %ss: %s"
            % (self.unit, self.log_["#non-significant %ss" % self.unit]),
            fontsize=12,
        )
        ax.text(0,
            0.0,
            "#silent %ss: %s" % (self.unit, self.log_["#silent %ss" % self.unit]),
            fontsize=12,
        )
        ax.axis("off")
        gs = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[16:26, 64:100])
        ax = fig.add_subplot(gs[0, 0])
        ax.text(0, 1.0, r"#Invalid cells: %s" % self.num_invalid_cells, fontsize=12)
        ax.text(0, 0.5, r"Essential dimension $\ell$: %s" % self.log_["ell"], fontsize=12)
        ax.text(0, 0, r"Elapsed time: %s" % self.elapsed_time, fontsize=12)
        ax.axis("off")
        #
        gs = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[34, 0])
        ax = fig.add_subplot(gs[0, 0])
        ax.text(0, 0.0, "Applicability", fontsize=16, fontweight="bold")
        ax.axis("off")
        #
        ps = 2
        gs = GridSpecFromSubplotSpec(
            nrows=1,
            ncols=2,
            width_ratios=[4, 1],
            subplot_spec=gs_master[37:85, 8:100],
            wspace=0.0,
        )
        ax0 = fig.add_subplot(gs[0, 0])
        x, y = np.mean(X_scaled, axis=0), norm_var
        idx_nonsig, idx_sig = y <= 1, y > 1
        if self.seq_target == "Multiome":
            ax0.scatter(
                x[idx_sig & (self.idx_atac == False)],
                y[idx_sig & (self.idx_atac == False)],
                color="b",
                s=ps,
                label="significant genes",
                zorder=2,
                marker="o",
                facecolor="None",
            )
            ax0.scatter(
                x[idx_sig & self.idx_atac],
                y[idx_sig & self.idx_atac],
                color="lightblue",
                s=ps,
                label="significant peaks",
                zorder=2,
                marker="o",
                facecolor="None",
            )
            ax0.scatter(
                x[idx_nonsig & (self.idx_atac == False)],
                y[idx_nonsig & (self.idx_atac == False)],
                color="r",
                s=ps,
                label="non-significant genes",
                zorder=3,
                marker="x",
            )
            ax0.scatter(
                x[idx_nonsig & self.idx_atac],
                y[idx_nonsig & self.idx_atac],
                color="orange",
                s=ps,
                label="non-significant peaks",
                zorder=3,
                marker="x",
            )
        else:
            ax0.scatter(
                x[idx_sig],
                y[idx_sig],
                color="b",
                s=ps,
                label="significant %s" % self.unit,
                zorder=2,
            )
            ax0.scatter(
                x[idx_nonsig],
                y[idx_nonsig],
                color="r",
                s=ps,
                label="non-significant %s" % self.unit,
                zorder=3,
            )
        # ax0.scatter(x[idx_sig],y[idx_sig],color='b',s=ps,label='significant %ss' % self.unit,zorder=2)
        # ax0.scatter(x[idx_nonsig],y[idx_nonsig],color='r',s=ps,label='non-significant %ss' % self.unit,zorder=3)
        ax0.axhline(1, color="gray", ls="--", lw=2, zorder=1)
        ax0.set_xscale("log")
        ax0.set_yscale("log")
        ax0.set_xlabel("Mean of scaled data", fontsize=14)
        ax0.set_ylabel("NVSN variance", fontsize=14)
        ax0.legend(
            loc="upper left",
            borderaxespad=0,
            fontsize=14,
            markerscale=5,
            handletextpad=0.0,
        ).get_frame().set_alpha(0)
        ylim = ax0.set_ylim()
        ax1 = fig.add_subplot(gs[0, 1])
        sns.kdeplot(y=np.log10(norm_var[norm_var > 0]), color="k", fill=True, ax=ax1)
        ax1.axhline(0, c="gray", ls="--", lw=2, zorder=1)
        ax1.axvline(0, c="k", ls="-", lw=1, zorder=1)
        ax1.set_ylim(np.log10(ax0.set_ylim()))
        ax1.tick_params(labelbottom=True, labelleft=False, bottom=True)
        ax1.set_xlabel("Density", fontsize=14)
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.tick_params(left=False)
        ax1.patch.set_alpha(0)
        x = np.linspace(ax1.set_ylim()[0], ax1.set_ylim()[1], 1000)
        dens = scipy.stats.gaussian_kde(np.log10(norm_var[norm_var > 0]))(x)
        peak_val = x[np.argmax(dens)]
        rate_low_var = np.sum(norm_var[norm_var > 0] < 0.90) / len(
            norm_var[norm_var > 0]
        )
        applicability = "Unknown"
        backcolor = "w"
        if (rate_low_var < 0.01) and (np.abs(peak_val) < 0.1):
            applicability = "Class A (strongly applicable)"
            backcolor = "lightgreen"
        elif rate_low_var < 0.01:
            applicability = "Class B (weakly applicable)"
            backcolor = "yellow"
        else:
            applicability = "Class C (inapplicabile)"
            backcolor = "tomato"
        ax0.text(
            0.99,
            0.975,
            applicability,
            va="top",
            ha="right",
            transform=ax0.transAxes,
            fontsize=14,
            backgroundcolor=backcolor,
        )
        #
        gs = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[100, 0])
        ax = fig.add_subplot(gs[0, 0])
        ax.text(
            0,
            0.0,
            "PC variance modification/elimination",
            fontsize=16,
            fontweight="bold",
        )
        ax.axis("off")
        #
        gs = GridSpecFromSubplotSpec(
            nrows=1, ncols=1, subplot_spec=gs_master[105:145, 8:80]
        )
        ps = 10
        # n_plot = n_EV if n_EV < 1000 else 1000
        # n_plot = self.recode_.ell if self.recode_.ell > 1000 else n_plot
        n_plot = len(plot_EV)
        ax = fig.add_subplot(gs[0, 0])
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        ax.scatter(
            np.arange(n_plot) + 1,
            plot_EV[:n_plot],
            color="lightblue",
            label="Original",
            marker="^",
            s=ps,
            zorder=1,
        )
        ax.scatter(
            np.arange(self.recode_.ell) + 1,
            plot_EV_mod[: self.recode_.ell],
            color="green",
            label="PC variance \nmodification",
            s=ps,
            zorder=2,
        )
        ax.scatter(
            np.arange(self.recode_.ell, n_plot) + 1,
            plot_EV_mod[self.recode_.ell : n_plot],
            color="orange",
            label="PC variance \nelimination",
            s=ps,
            zorder=2,
        )
        ax.axhline(0, color="gray", ls="--", zorder=-10)
        ax.axvline(self.recode_.ell, color="gray", ls="--")
        ax.text(
            self.recode_.ell * 1.1,
            0.3,
            r"$\ell$=%d" % int(self.recode_.ell),
            color="k",
            fontsize=16,
            ha="left",
        )
        ax.set_xlabel("PC", fontsize=fs_label)
        ax.set_ylabel("PC variance (eigenvalue)", fontsize=fs_label)
        ax.set_yscale("symlog")
        ax.set_xlim([-5, n_plot + 5])
        ax.set_ylim([-0.5, max(plot_EV) * 1.5])
        ax.legend(
            bbox_to_anchor=(1.00, 1),
            loc="upper left",
            borderaxespad=0,
            fontsize=12,
            markerscale=2,
            handletextpad=0.0,
        ).get_frame().set_alpha(0)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        #
        gs = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[155, 0])
        ax = fig.add_subplot(gs[0, 0])
        ax.text(
            0,
            0.0,
            "Mean-variance plot (log-normalized data)",
            fontsize=16,
            fontweight="bold",
        )
        ax.axis("off")
        #
        titles = ("Original", "RECODE")
        ps = 1
        fs_title = 14
        gs = GridSpecFromSubplotSpec(
            nrows=1, ncols=2, subplot_spec=gs_master[163:200, 5:100]
        )
        ax0 = fig.add_subplot(gs[0, 0])
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        x, y = np.mean(X_ss_log, axis=0), np.var(X_ss_log, axis=0, ddof=1)
        if self.seq_target == "Multiome":
            ax0.scatter(
                x[idx_sig & (self.idx_atac == False)],
                y[idx_sig & (self.idx_atac == False)],
                color="b",
                s=ps,
                label="significant genes",
                zorder=2,
                marker="o",
                facecolor="None",
            )
            ax0.scatter(
                x[idx_sig & self.idx_atac],
                y[idx_sig & self.idx_atac],
                color="lightblue",
                s=ps,
                label="significant peaks",
                zorder=2,
                marker="o",
                facecolor="None",
            )
            ax0.scatter(
                x[idx_nonsig & (self.idx_atac == False)],
                y[idx_nonsig & (self.idx_atac == False)],
                color="r",
                s=ps,
                label="non-significant genes",
                zorder=3,
                marker="x",
            )
            ax0.scatter(
                x[idx_nonsig & self.idx_atac],
                y[idx_nonsig & self.idx_atac],
                color="orange",
                s=ps,
                label="non-significant peaks",
                zorder=3,
                marker="x",
            )
        else:
            ax0.scatter(
                x[idx_sig],
                y[idx_sig],
                color="b",
                s=ps,
                label="significant %s" % self.unit,
                zorder=2,
            )
            ax0.scatter(
                x[idx_nonsig],
                y[idx_nonsig],
                color="r",
                s=ps,
                label="non-significant %s" % self.unit,
                zorder=3,
            )
        ax0.axhline(0, color="gray", ls="--", lw=2, zorder=1)
        ax0.set_xlabel("Mean", fontsize=fs_label)
        ax0.set_ylabel("Variance", fontsize=fs_label)
        ax0.set_title(titles[0], fontsize=fs_title)
        ylim = [-0.05 * np.percentile(y, 99.9), 1.05 * np.percentile(y, 99.9)]
        ax0.set_ylim(ylim)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        ax1 = fig.add_subplot(gs[0, 1])
        x, y = np.mean(X_RECODE_ss_log, axis=0), np.var(X_RECODE_ss_log, axis=0, ddof=1)
        if self.seq_target == "Multiome":
            ax1.scatter(
                x[idx_sig & (self.idx_atac == False)],
                y[idx_sig & (self.idx_atac == False)],
                color="b",
                s=ps,
                label="significant genes",
                zorder=2,
                marker="o",
                facecolor="None",
            )
            ax1.scatter(
                x[idx_sig & self.idx_atac],
                y[idx_sig & self.idx_atac],
                color="lightblue",
                s=ps,
                label="significant peaks",
                zorder=2,
                marker="o",
                facecolor="None",
            )
            ax1.scatter(
                x[idx_nonsig & (self.idx_atac == False)],
                y[idx_nonsig & (self.idx_atac == False)],
                color="r",
                s=ps,
                label="non-significant genes",
                zorder=3,
                marker="x",
            )
            ax1.scatter(
                x[idx_nonsig & self.idx_atac],
                y[idx_nonsig & self.idx_atac],
                color="orange",
                s=ps,
                label="non-significant peaks",
                zorder=3,
                marker="x",
            )
        else:
            ax1.scatter(
                x[idx_sig],
                y[idx_sig],
                color="b",
                s=ps,
                label="significant %s" % self.unit,
                zorder=2,
            )
            ax1.scatter(
                x[idx_nonsig],
                y[idx_nonsig],
                color="r",
                s=ps,
                label="non-significant %s" % self.unit,
                zorder=3,
            )
        ax1.set_ylim(ylim)
        ax1.axhline(0, color="gray", ls="--", lw=2, zorder=1)
        ax1.set_xlabel("Mean", fontsize=fs_label)
        ax1.set_ylabel("Variance", fontsize=fs_label)
        ax1.set_title(titles[1], fontsize=fs_title)
        ax1.legend(
            loc="upper right",
            borderaxespad=0,
            fontsize=10,
            markerscale=3,
            handletextpad=0.0,
        ).get_frame().set_alpha(0)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        if save:
            plt.savefig(
                "%s.%s" % (save_filename, save_format), dpi=dpi, bbox_inches="tight"
            )
        if show:
            plt.show()

    def plot_original_data(
        self,
        title="",
        figsize=(7, 5),
        save=False,
        save_filename="original_data",
        save_format="png",
        dpi=None,
        show=True,
    ):
        """
        Plot noise variance for each features.

        Parameters
        ----------
        title : str, default=''
                Figure title.

        figsize : 2-tuple of floats, default=(7,5)
                Figure dimension ``(width, height)`` in inches.

        ps : float, default=10,
                Point size.

        save : bool, default=False
                If True, save the figure.

        save_filename : str, default='original_data',
                File name (path) of save figure.

        save_format : {'png', 'pdf', 'svg'}, default= 'png',
                File format of save figure.

        dpi: float or None, default=None
                Dots per inch.
        """
        ps = 2
        fs_title = 16
        fs_label = 14
        X_nUMI = np.sum(self.X_temp, axis=1)
        # X_scaled = (self.X_temp.T / X_nUMI).T
        X_scaled = self.X_temp / X_nUMI[:,np.newaxis]
        # X_scaled_mean = np.mean(X_scaled, axis=0)
        noise_var = np.mean(self.X_temp / X_nUMI[:,np.newaxis] / X_nUMI[:,np.newaxis], axis=0)
        noise_var[noise_var == 0] = 1
        # X_norm = (X_scaled - np.mean(X_scaled, axis=0)) / np.sqrt(noise_var)
        fig, ax = plt.subplots(figsize=figsize)
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        idx_sort = np.argsort(np.mean(X_scaled, axis=0))
        x = np.arange(X_scaled.shape[1])
        y1 = np.var(X_scaled, axis=0, ddof=1)[idx_sort]
        y2 = noise_var[idx_sort]
        plt1 = ax.scatter(x, y1, color="k", s=ps, label="Original", zorder=1)
        plt2 = ax.scatter(x, y2, color="r", s=ps, label="Noise", zorder=2, marker="x")
        ax.axhline(0, color="gray", ls="--", lw=2, zorder=1)
        ax.set_ylim([min(min(y1), min(y2)) * 0.5, max(max(y1), max(y2))])
        ax.set_xlabel(self.Unit, fontsize=fs_label)
        ax.set_ylabel("Variance", fontsize=fs_label)
        ax.set_yscale("log")
        ax.legend(
            loc="upper left",
            borderaxespad=0,
            fontsize=14,
            markerscale=5,
            handletextpad=0.0,
        ).get_frame().set_alpha(0)
        plt1.set_alpha(0.1)
        plt2.set_alpha(0.1)
        ax.set_title(title, fontsize=fs_title)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        if save:
            plt.savefig(
                "%s.%s" % (save_filename, save_format), dpi=dpi, bbox_inches="tight"
            )
        if show:
            plt.show()

    def plot_normalized_data(
        self,
        title="Normalized data",
        figsize=(7, 5),
        save=False,
        save_filename="RECODE_noise_variance",
        save_format="png",
        dpi=None,
        show=True,
    ):
        """
        Plot the transformed data by the noise variance-stabilizing normalization.

        Parameters
        ----------
        title : str, default=''
                Figure title.

        figsize : 2-tuple of floats, default=(7,5)
                Figure dimension ``(width, height)`` in inches.

        ps : float, default=10,
                Point size.

        save : bool, default=False
                If True, save the figure.

        save_filename : str, default= 'noise_variance',
                File name (path) of save figure.

        save_format : {'png', 'pdf', 'svg'}, default= 'png',
                File format of save figure.

        dpi: float or None, default=None
                Dots per inch.
        """
        ps = 1
        fs_title = 16
        fs_label = 14
        X_nUMI = np.sum(self.X_temp, axis=1)
        # X_scaled = (self.X_temp.T / X_nUMI).T
        X_scaled = self.X_temp / X_nUMI[:,np.newaxis]
        # X_scaled_mean = np.mean(X_scaled, axis=0)
        # noise_var = np.mean(self.X_temp.T / X_nUMI / X_nUMI, axis=1)
        noise_var = np.mean(self.X_temp / X_nUMI[:,np.newaxis] / X_nUMI[:,np.newaxis], axis=0)
        noise_var[noise_var == 0] = 1
        X_norm = (X_scaled - np.mean(X_scaled, axis=0)) / np.sqrt(noise_var)
        fig, ax = plt.subplots(figsize=figsize)
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        idx_sort = np.argsort(np.mean(X_scaled, axis=0))
        x = np.arange(X_scaled.shape[1])
        ax.scatter(
            x, np.var(X_norm, axis=0, ddof=1)[idx_sort], color="k", s=ps, zorder=2
        )
        ax.axhline(1, color="r", ls="--")
        ax.set_xlabel(self.Unit, fontsize=fs_label)
        ax.set_ylabel("NVSN variance", fontsize=fs_label)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=fs_title)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        if save:
            plt.savefig(
                "%s.%s" % (save_filename, save_format), dpi=dpi, bbox_inches="tight"
            )
        if show:
            plt.show()

    def plot_projected_data(
        self,
        title="Projected data",
        figsize=(7, 5),
        save=False,
        save_filename="RECODE_noise_variance",
        save_format="png",
        dpi=None,
        show=True,
    ):
        """
        Plot projected data.

        Parameters
        ----------
        title : str, default=''
                Figure title.

        figsize : 2-tuple of floats, default=(7,5)
                Figure dimension ``(width, height)`` in inches.

        ps : float, default=10,
                Point size.

        save : bool, default=False
                If True, save the figure.

        save_filename : str, default= 'noise_variance',
                File name (path) of save figure.

        save_format : {'png', 'pdf', 'svg'}, default= 'png',
                File format of save figure.

        dpi: float or None, default=None
                Dots per inch.
        """
        ps = 20
        fs_title = 16
        fs_label = 14

        plot_EV = self.recode_.PCA_Ev[self.recode_.PCA_Ev > 0]
        n_EV = len(plot_EV)
        # X_scaled = (self.X_temp.T / np.sum(self.X_temp, axis=1)).T
        X_scaled = self.X_temp / np.sum(self.X_temp, axis=1)[:,np.newaxis]
        # idx_sort = np.argsort(np.mean(X_scaled, axis=0))
        fig, ax = plt.subplots(figsize=figsize)
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        idx_sort = np.argsort(np.mean(X_scaled, axis=0))
        x = np.arange(X_scaled.shape[1])
        ax.scatter(
            np.arange(n_EV) + 1, plot_EV, color="k", label="Original", s=ps, zorder=1
        )
        ax.set_xlabel("PC", fontsize=fs_label)
        ax.set_ylabel("PC variance (eigenvalue)", fontsize=fs_label)
        ax.set_yscale("symlog")
        ax.set_ylim([0, max(plot_EV) * 1.5])
        ax.set_title(title, fontsize=fs_title)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        if save:
            plt.savefig(
                "%s.%s" % (save_filename, save_format), dpi=dpi, bbox_inches="tight"
            )
        if show:
            plt.show()

    def plot_variance_modified_data(
        self,
        title="Variance-modified data",
        figsize=(7, 5),
        save=False,
        save_filename="RECODE_noise_variance",
        save_format="png",
        dpi=None,
        show=True,
    ):
        """
        Plot varainces (eigenvalues) of the variance-modified data.

        Parameters
        ----------
        title : str, default=''
                Figure title.

        figsize : 2-tuple of floats, default=(7,5)
                Figure dimension ``(width, height)`` in inches.

        ps : float, default=10,
                Point size.

        save : bool, default=False
                If True, save the figure.

        save_filename : str, default= 'noise_variance',
                File name (path) of save figure.

        save_format : {'png', 'pdf', 'svg'}, default= 'png',
                File format of save figure.

        dpi: float or None, default=None
                Dots per inch.
        """
        ps = 20
        fs_title = 16
        fs_label = 14

        plot_EV = self.recode_.PCA_Ev[self.recode_.PCA_Ev > 0]
        n_EV = len(plot_EV)
        plot_EV_mod = np.zeros(n_EV)
        plot_EV_mod[: self.recode_.ell] = self.recode_.PCA_Ev_NRM[: self.recode_.ell]
        # X_scaled = (self.X_temp.T / np.sum(self.X_temp, axis=1)).T
        X_scaled = self.X_temp / np.sum(self.X_temp, axis=1)[:,np.newaxis]
        # idx_sort = np.argsort(np.mean(X_scaled, axis=0))
        fig, ax = plt.subplots(figsize=figsize)
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        idx_sort = np.argsort(np.mean(X_scaled, axis=0))
        x = np.arange(X_scaled.shape[1])
        ax.scatter(
            np.arange(n_EV) + 1,
            plot_EV,
            color="lightblue",
            label="Original",
            marker="^",
            s=ps,
            zorder=1,
        )
        ax.scatter(
            np.arange(n_EV) + 1,
            plot_EV_mod,
            color="k",
            label="Modified",
            s=ps,
            zorder=2,
        )
        ax.axvline(self.recode_.ell, color="gray", ls="--")
        ax.text(
            self.recode_.ell * 1.1,
            0.2,
            r"$\ell$=%d" % self.recode_.ell,
            color="k",
            fontsize=16,
            ha="left",
        )
        ax.set_xlabel("PC", fontsize=fs_label)
        ax.set_ylabel("PC variance (eigenvalue)", fontsize=fs_label)
        # ax.set_xscale('log')
        ax.set_yscale("symlog")
        ax.set_ylim([0, max(plot_EV) * 1.5])
        ax.legend(
            loc="upper right",
            borderaxespad=0,
            fontsize=14,
            markerscale=2,
            handletextpad=0.0,
        ).get_frame().set_alpha(0)
        ax.set_title(title, fontsize=fs_title)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        if save:
            plt.savefig(
                "%s.%s" % (save_filename, save_format), dpi=dpi, bbox_inches="tight"
            )
        if show:
            plt.show()

    def plot_denoised_data(
        self,
        title="",
        figsize=(7, 5),
        save=False,
        save_filename="RECODE_denoised_data",
        save_format="png",
        dpi=None,
        show=True,
    ):
        """
        Plot variances of the denoised data.

        Parameters
        ----------
        title : str, default=''
                Figure title.

        figsize : 2-tuple of floats, default=(7,5)
                Figure dimension ``(width, height)`` in inches.

        ps : float, default=10,
                Point size.

        save : bool, default=False
                If True, save the figure.

        save_filename : str, default= 'noise_variance',
                File name (path) of save figure.

        save_format : {'png', 'pdf', 'svg'}, default= 'png',
                File format of save figure.

        dpi: float or None, default=None
                Dots per inch.
        """
        ps = 2
        fs_title = 16
        fs_label = 14
        # X_scaled = (self.X_temp.T / np.sum(self.X_temp, axis=1)).T
        # X_RECODE_scaled = (
        #     self.X_RECODE[:, self.idx_nonsilent].T / np.sum(self.X_RECODE, axis=1)
        # ).T
        X_scaled = self.X_temp / np.sum(self.X_temp, axis=1)[:,np.newaxis]
        X_RECODE_scaled = self.X_RECODE[:, self.idx_nonsilent] / np.sum(self.X_RECODE, axis=1)[:,np.newaxis]
        idx_sort = np.argsort(np.mean(X_scaled, axis=0))
        fig, ax = plt.subplots(figsize=figsize)
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        x = np.arange(X_scaled.shape[1])
        y1 = np.var(X_scaled, axis=0, ddof=1)[idx_sort]
        y2 = np.var(X_RECODE_scaled, axis=0, ddof=1)[idx_sort]
        plt1 = ax.scatter(
            x, y1, color="lightblue", s=ps, label="Original", zorder=1, marker="^"
        )
        plt2 = ax.scatter(
            x, y2, color="k", s=ps, label="Denoised", zorder=2, marker="o"
        )
        ax.set_ylim([min(min(y1), min(y2)) * 0.5, max(max(y1), max(y2))])
        ax.set_yscale("log")
        ax.set_xlabel(self.Unit, fontsize=fs_label)
        ax.set_ylabel("Variance", fontsize=fs_label)
        ax.legend(
            loc="upper left",
            borderaxespad=0,
            fontsize=14,
            markerscale=7,
            handletextpad=0.0,
        ).get_frame().set_alpha(0)
        plt1.set_alpha(0.05)
        plt2.set_alpha(0.1)
        ax.set_title(title, fontsize=fs_title)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        if save:
            plt.savefig(
                "%s.%s" % (save_filename, save_format), dpi=dpi, bbox_inches="tight"
            )
        if show:
            plt.show()

    def plot_mean_variance(
        self,
        titles=("Original", "RECODE"),
        figsize=(7, 5),
        ps=2,
        target_sum="median",
        save=False,
        save_filename="plot_mean_variance",
        save_format="png",
        dpi=None,
        show=True,
    ):
        """
        Plot mean vs variance of features for log-normalized data

        Parameters
        ----------
        titles : tuple, default=('Original','RECODE')
                Figure title.

        figsize : 2-tuple of floats, default=(7,5)
                Figure dimension ``(width, height)`` in inches.

        ps : float, default=10,
                Point size.

        target_sum : float or {'median','mean'}, default='median',
                Size factor (total count constant of each cell before the log-normalization).

        save : bool, default=False
                If True, save the figure.

        save_filename : str, default= 'check_applicability',
                File name (path) of save figure.

        save_format : {'png', 'pdf', 'svg'}, default= 'png',
                File format of save figure.

        dpi: float or None, default=None
                Dots per inch.
        """
        fs_label = 14
        fs_title = 14
        if target_sum == "median":
            target_sum = np.median(np.sum(self.X_trans, axis=1))
            target_sum_RECODE = np.median(np.sum(self.X_RECODE, axis=1))
        elif target_sum == "mean":
            target_sum = np.mean(np.sum(self.X_trans, axis=1))
            target_sum_RECODE = np.mean(np.sum(self.X_REECODE, axis=1))
        elif (type(target_sum) == int) | (type(target_sum) == float):
            target_sum_RECODE = target_sum
        else:
            target_sum = np.median(np.sum(self.X_trans, axis=1))
            target_sum_RECODE = np.median(np.sum(self.X_RECODE, axis=1))
        # X_ss_log = np.log(
        #     target_sum
        #     * (self.X_trans[:, self.idx_nonsilent].T / np.sum(self.X_trans, axis=1)).T
        #     + 1
        # )
        # X_RECODE_ss_log = np.log(
        #     target_sum_RECODE
        #     * (self.X_RECODE[:, self.idx_nonsilent].T / np.sum(self.X_RECODE, axis=1)).T
        #     + 1
        # )
        X_ss_log = np.log(target_sum * self.X_trans[:, self.idx_nonsilent] / np.sum(self.X_trans, axis=1)[:,np.newaxis]+ 1)
        X_RECODE_ss_log = np.log(target_sum_RECODE * self.X_RECODE[:, self.idx_nonsilent] / np.sum(self.X_RECODE, axis=1)[:,np.newaxis] + 1)
        fig, ax0 = plt.subplots(figsize=figsize)
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        x, y = np.mean(X_ss_log, axis=0), np.var(X_ss_log, axis=0, ddof=1)
        ax0.scatter(
            x, y, color="b", s=ps, label="significant %ss" % self.unit, zorder=2
        )
        ax0.axhline(0, color="gray", ls="--", lw=2, zorder=1)
        ax0.set_xlabel("Mean of log-scaled data", fontsize=fs_label)
        ax0.set_ylabel("Variance of log-scaled data", fontsize=fs_label)
        ax0.set_title(titles[0], fontsize=fs_title)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        if save:
            plt.savefig(
                "%s_Original.%s" % (save_filename, save_format),
                dpi=dpi,
                bbox_inches="tight",
            )
        fig, ax1 = plt.subplots(figsize=figsize)
        x, y = np.mean(X_RECODE_ss_log, axis=0), np.var(X_RECODE_ss_log, axis=0, ddof=1)
        ax1.scatter(
            x, y, color="b", s=ps, label="significant %ss" % self.unit, zorder=2
        )
        # ax1.set_ylim(ax0.set_ylim())
        ax1.axhline(0, color="gray", ls="--", lw=2, zorder=1)
        ax1.set_xlabel("Mean of log-scaled data", fontsize=fs_label)
        ax1.set_ylabel("Variance of log-scaled data", fontsize=fs_label)
        ax1.set_title(titles[1], fontsize=fs_title)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        if save:
            plt.savefig(
                "%s_RECODE.%s" % (save_filename, save_format),
                dpi=dpi,
                bbox_inches="tight",
            )
        if show:
            plt.show()

    def plot_mean_cv(
        self,
        titles=("Original", "RECODE"),
        figsize=(7, 5),
        ps=2,
        save=False,
        save_filename="plot_mean_cv",
        save_format="png",
        dpi=None,
        show_features=False,
        n_show_features=10,
        cut_detect_rate=0.005,
        index=None,
        show=True,
    ):
        """
        Plot mean vs variance of features for log-normalized data

        Parameters
        ----------
        title : str, default=''
                Figure title.

        figsize : 2-tuple of floats, default=(7,5)
                Figure dimension ``(width, height)`` in inches.

        ps : float, default=10,
                Point size.

        save : bool, default=False
                If True, save the figure.

        save_filename : str, default= 'check_applicability',
                File name (path) of save figure.

        save_format : {'png', 'pdf', 'svg'}, default= 'png',
                File format of save figure.

        dpi : float or None, default=None
                Dots per inch.

        show_features : float or None, default=False,
                If True

        n_show_features : float, default=10,

        cut_detect_rate : float, default=0.005,

        index : array-like of shape (n_features,) or None, default=None,


        """
        fs_label = 14
        fs_title = 14
        # X_ss = (
        #     np.median(np.sum(self.X_trans[:, self.idx_nonsilent], axis=1))
        #     * self.X_trans[:, self.idx_nonsilent].T
        #     / np.sum(self.X_trans[:, self.idx_nonsilent], axis=1)
        # ).T
        X_ss = (np.median(np.sum(self.X_trans[:, self.idx_nonsilent], axis=1)) * self.X_trans[:, self.idx_nonsilent] / np.sum(self.X_trans[:, self.idx_nonsilent], axis=1)[:,np.newaxis])
        fig, ax0 = plt.subplots(figsize=figsize)
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        x = np.mean(X_ss, axis=0)
        cv = np.std(X_ss, axis=0) / np.mean(X_ss, axis=0)
        ax0.scatter(x, cv, color="b", s=ps, zorder=2)
        ax0.axhline(0, color="gray", ls="--", lw=2, zorder=1)
        ax0.set_xscale("log")
        ax0.set_xlabel("Mean", fontsize=fs_label)
        ax0.set_ylabel("Coefficient of variation", fontsize=fs_label)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        ax0.set_title(titles[0], fontsize=fs_title)
        if save:
            plt.savefig(
                "%s_Original.%s" % (save_filename, save_format),
                dpi=dpi,
                bbox_inches="tight",
            )

        # X_RECODE_ss = (
        #     np.median(np.sum(self.X_RECODE[:, self.idx_nonsilent], axis=1))
        #     * self.X_RECODE[:, self.idx_nonsilent].T
        #     / np.sum(self.X_RECODE[:, self.idx_nonsilent], axis=1)
        # ).T
        X_RECODE_ss = np.median(np.sum(self.X_RECODE[:, self.idx_nonsilent], axis=1)) * self.X_RECODE[:, self.idx_nonsilent] / np.sum(self.X_RECODE[:, self.idx_nonsilent], axis=1)[:,np.newaxis]
        fig, ax1 = plt.subplots(figsize=figsize)
        x = np.mean(X_RECODE_ss, axis=0)
        cv = np.std(X_RECODE_ss, axis=0) / np.mean(X_RECODE_ss, axis=0)
        # ax1.set_ylim(ax0.set_ylim())
        ax1.axhline(0, color="gray", ls="--", lw=2, zorder=1)
        ax1.set_xscale("log")
        ax1.set_xlabel("Mean", fontsize=fs_label)
        ax1.set_ylabel("Coefficient of variation", fontsize=fs_label)
        ax1.set_title(titles[1], fontsize=fs_title)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)

        if show_features:
            if len(index) != self.X.shape[1]:
                self.logger.warning(
                    "Warning: no index opotion or length of index did not fit X.shape[1]. Use feature numbers"
                )
                index = np.arange(self.X.shape[1]) + 1
            detect_rate = (
                np.sum(np.where(self.X > 0, 1, 0), axis=0)[self.idx_nonsilent]
                / self.X.shape[0]
            )
            idx_detect_rate_n = detect_rate <= cut_detect_rate
            idx_detect_rate_p = detect_rate > cut_detect_rate
            ax1.scatter(
                x[idx_detect_rate_n],
                cv[idx_detect_rate_n],
                color="gray",
                s=ps,
                label="detection rate <= {:.2%}".format(cut_detect_rate),
                alpha=0.5,
            )
            ax1.scatter(
                x[idx_detect_rate_p],
                cv[idx_detect_rate_p],
                color="b",
                s=ps,
                label="detection rate > {:.2%}".format(cut_detect_rate),
                alpha=0.5,
            )
            ax1.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=2,
                fontsize=12,
                markerscale=2,
            )
            idx_rank_cv = np.argsort(cv[idx_detect_rate_p])[::-1]
            texts = [
                plt.text(
                    x[idx_detect_rate_p][idx_rank_cv[i]],
                    cv[idx_detect_rate_p][idx_rank_cv[i]],
                    index[self.idx_nonsilent][idx_detect_rate_p][idx_rank_cv[i]],
                    color="red",
                )
                for i in range(n_show_features)
            ]
            try:
                import adjustText
            except ImportError:
                raise ImportError("\nplease install adjustText:\n\n\tpip install adjustText")
            adjustText.adjust_text(texts, arrowprops=dict(arrowstyle="->", color="k"))
        else:
            ax1.scatter(x, cv, color="b", s=ps, zorder=2)

        if save:
            plt.savefig(
                "%s_RECODE.%s" % (save_filename, save_format),
                dpi=dpi,
                bbox_inches="tight",
            )
        if show:
            plt.show()

    def plot_ATAC_preprocessing(
        self,
        title="ATAC preprocessing",
        figsize=(7, 5),
        ps=10,
        save=False,
        save_filename="plot_ATAC_preprocessing",
        save_format="png",
        dpi=None,
        show=True,
    ):
        """
        Plot the number of values in scATAC-seq data matrix with and without preprocessing (odd-even stabilization).

        Parameters
        ----------
        title : str, default='ATAC preprocessing'
                Figure title.

        figsize : 2-tuple of floats, default=(7,5)
                Figure dimension ``(width, height)`` in inches.

        ps : float, default=10,
                Point size.

        save : bool, default=False
                If True, save the figure.

        save_filename : str, default= 'plot_ATAC_preprocessing',
                File name (path) of save figure.

        save_format : {'png', 'pdf', 'svg'}, default= 'png',
                File format of save figure.

        dpi: float or None, default=None
                Dots per inch.
        """
        if self.seq_target != "ATAC":
            self.logger.warning(
                "Error: plot_ATAC_preprocessing is an option of scATAC-seq data"
            )
            return
        ps = 1
        fs_title = 16
        fs_label = 14
        fs_legend = 14
        val, count = np.unique(self.X_trans, return_counts=True)
        idx_even = np.empty(len(val), dtype=bool)
        idx_odd = np.empty(len(val), dtype=bool)
        for i in range(len(val)):
            if i > 0 and i % 2 == 0:
                idx_even[i] = True
            else:
                idx_even[i] = False
            if i > 0 and i % 2 == 1:
                idx_odd[i] = True
            else:
                idx_odd[i] = False
        plt.figure(figsize=figsize)
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.plot(
            val[1:],
            count[1:],
            color="lightblue",
            zorder=1,
            marker="^",
            label="Original",
        )
        val, count = np.unique(self.X_temp, return_counts=True)
        plt.plot(
            val[1:], count[1:], color="gray", marker="o", label="Preprpcessed", zorder=3
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("value", fontsize=fs_label)
        plt.ylabel("count", fontsize=fs_label)
        plt.title(title, fontsize=fs_title)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.legend(fontsize=fs_legend)
        if save:
            plt.savefig(
                "%s.%s" % (save_filename, save_format), dpi=dpi, bbox_inches="tight"
            )
        if show:
            plt.show()


class RECODE_core:
    def __init__(
        self,
        method="variance",
        variance_estimate=True,
        fast_algorithm=True,
        fast_algorithm_ell_ub=1000,
        ell_manual=10,
        ell_min=3,
        version=1,
        random_state=0,
        verbose=True,
    ):
        """
        The core part of RECODE (for non-randam sampling data).

        Parameters
        ----------
        method : {'variance','manual'}
                If 'variance', regular variance-based algorithm.
                If 'manual', parameter ell, which identifies essential and noise parts in the PCA space, is manually set. The manual parameter is given by ``ell_manual``.

        variance_estimate : boolean, default=True
                If True and ``method='variance'``, the parameter estimation method will be done.

        fast_algorithm : boolean, default=True
                If True, the fast algorithm is done. The upper bound of essential dimension ell is set in ``fast_algorithm_ell_ub``.

        fast_algorithm_ell_ub : int, default=1000
                Upper bound of parameter ell for the fast algorithm. Must be of range [1, infinity).

        ell_manual : int, default=10
                Manual essential dimension ell computed by ``method='manual'``. Must be of range [1, infinity).

        ell_min : int, default=3
                Minimam value of essential dimension ell

        version : int default='1'
                Version of RECODE.

        """
        self.method = method
        self.variance_estimate = variance_estimate
        self.fast_algorithm = fast_algorithm
        self.fast_algorithm_ell_ub = fast_algorithm_ell_ub
        self.ell_manual = ell_manual
        self.ell_min = ell_min
        self.fit_idx = False
        self.version = version
        self.RECODE_done = False
        self.random_state = random_state
        self.verbose = verbose
        self.logger = logging.getLogger("argument checking")
        if verbose:
            self.logger.setLevel(logging.WARNING)
        else:
            self.logger.setLevel(logging.ERROR)

    def _noise_reductor(self, X, L, U, Xmean, ell, version=1, return_ess=False):
        if version == 2 and self.RECODE_done == False:
            U_ell = U[:ell, :]
            L_ell = L[:ell, :ell]
            for i in range(ell):
                idx_order = np.argsort(U[i] ** 2)[::-1]
                idx_sparce = np.sort(U[i] ** 2)[::-1].cumsum() > L_ell[i, i]**2
                U_ell[i, idx_order[idx_sparce]] = 0
                if np.sqrt(np.sum(U_ell[i] ** 2)) >0:
                    U_ell[i] = U_ell[i] / np.sqrt(np.sum(U_ell[i] ** 2))
            X_ess = np.dot(np.dot(X - Xmean, U_ell.T), L_ell)
            X_recode = np.dot(X_ess, U_ell) + Xmean
        else:
            U_ell = U[:ell, :]
            L_ell = L[:ell, :ell]
            X_ess = np.dot(np.dot(X - Xmean, U_ell.T), L_ell)
            X_recode = np.dot(X_ess, U_ell) + Xmean
        if return_ess:
            return X_recode, X_ess, U_ell, Xmean
        else:
            return X_recode

    def _noise_reduct_param(self, X, delta=0.05):
        comp = max(np.sum(self.PCA_Ev_NRM > delta * self.PCA_Ev_NRM[0]), 3)
        self.ell = min(self.ell_max, comp)
        self.X_RECODE = self._noise_reductor(
            X, self.L, self.U, self.X_mean, self.ell, self.version, self.TO_CR
        )
        return self.X_RECODE

    def _noise_reduct_noise_var(self, X, noise_var=1):
        X_RECODE = self._noise_reductor(
            X, self.L, self.U, self.X_mean, self.ell, self.version, self.TO_CR
        )

        return X_RECODE

    def _noise_var_est(self, X, cut_low_exp=1.0e-10):
        n, d = X.shape
        X_var = np.var(X, axis=0, ddof=1)
        idx_var_p = np.where(X_var > cut_low_exp)[0]
        X_var_sub = X_var[idx_var_p]
        X_var_min = np.min(X_var_sub) - 1.0e-10
        X_var_max = np.max(X_var_sub) + 1.0e-10
        X_var_range = X_var_max - X_var_min

        div_max = 1000
        num_div_max = int(min(0.1 * d, div_max))
        error = np.empty(num_div_max)
        for i in range(num_div_max):
            num_div = i + 1
            delta = X_var_range / num_div
            k = np.empty([num_div], dtype=int)
            for j in range(num_div):
                div_min = j * delta + X_var_min
                div_max = (j + 1) * delta + X_var_min
                k[j] = len(np.where((X_var_sub < div_max) & (X_var_sub > div_min))[0])
            error[i] = (2 * np.mean(k) - np.var(k)) / delta / delta

        opt_div = int(np.argmin(error) + 1)

        k = np.empty([opt_div], dtype=int)
        delta = X_var_range / opt_div
        for j in range(opt_div):
            div_min = j * delta + X_var_min
            div_max = (j + 1) * delta + X_var_min
            k[j] = len(np.where((X_var_sub <= div_max) & (X_var_sub > div_min))[0])
        idx_k_max = np.argmax(k)
        div_min = idx_k_max * delta + X_var_min
        div_max = (idx_k_max + 1) * delta + X_var_min
        idx_set_k_max = np.where((X_var_sub < div_max) & (X_var_sub > div_min))[0]
        var = np.mean(X_var_sub[idx_set_k_max])
        return var

    def fit(self, X):
        """
        Create the transformation using X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features).
                Training data matrix, where ``n_samples`` is the number of samples
                and ``n_features`` is the number of features.
        Returns
        -------
        self : object
                Returns the instance itself.
        """
        n, d = X.shape
        n_pca = min(n - 1, d)
        if self.fast_algorithm:
            n_pca = min(n_pca, self.fast_algorithm_ell_ub)
        X_mean = np.mean(X, axis=0)

        if n > d:
            svd = sklearn.decomposition.TruncatedSVD(
                n_components=n_pca, random_state=self.random_state
            ).fit(X - X_mean)
            SVD_Sv = svd.singular_values_
            PCA_Ev = (SVD_Sv**2) / (n - 1)
            self.U = svd.components_
        else:
            SD = (X - X_mean) @ (X - X_mean).T / (n - 1)
            svd = sklearn.decomposition.TruncatedSVD(
                n_components=n_pca, random_state=self.random_state
            ).fit(SD)
            PCA_Ev = svd.singular_values_
            self.U = (svd.components_.T / np.sqrt(PCA_Ev)).T @ (X - X_mean) / np.sqrt(n - 1)

        PCA_Ev_sum_all = np.sum(np.var(X, axis=0, ddof=1))
        PCA_Ev_NRM = np.array(PCA_Ev, dtype=float)
        PCA_Ev_sum_diff = PCA_Ev_sum_all - np.sum(PCA_Ev)
        n_Ev_all = min(n, d)
        PCA_Ev_NRM = np.array(
            [
                PCA_Ev[i]
                - (np.sum(PCA_Ev[i + 1 :]) + PCA_Ev_sum_diff) / (n_Ev_all - i - 1)
                for i in range(len(PCA_Ev_NRM) - 1)
            ]
        )
        PCA_Ev_NRM = np.append(PCA_Ev_NRM, 0)
        PCA_CCR = (PCA_Ev / PCA_Ev_sum_all).cumsum()
        PCA_CCR_NRM = (PCA_Ev_NRM / np.sum(PCA_Ev_NRM)).cumsum()
        PCA_Ev_sum_diff = PCA_Ev_sum_all - np.sum(PCA_Ev)
        PCA_Ev_sum = (
            np.array([np.sum(PCA_Ev[i:]) for i in range(n_pca)]) + PCA_Ev_sum_diff
        )
        X_var = np.var(X, axis=0, ddof=1)
        dim = np.sum(X_var > 0)
        noise_var = 1
        if self.variance_estimate:
            noise_var = self._noise_var_est(X)
        thrshold = (dim - np.arange(n_pca)) * noise_var
        if np.sum(PCA_Ev_sum - thrshold < 0) == 0:
            self.logger.warning(
                "Acceleration error: the optimal value of ell is larger than fast_algorithm_ell_ub. Set larger fast_algorithm_ell_ub than %d or 'fast_algorithm=False'"
                % self.fast_algorithm_ell_ub
            )
            comp = n_pca-1
        else:
            comp = np.min(np.arange(n_pca)[PCA_Ev_sum - thrshold < 0])
        self.ell_max = np.min([n-1, d, len(PCA_CCR), np.sum(PCA_Ev > 1.0e-10)])
        self.ell = comp
        if self.ell > self.ell_max:
            self.ell = self.ell_max
        if self.ell < self.ell_min:
            self.ell = self.ell_min
        self.TO_CR = PCA_CCR[self.ell]
        self.TO_CR_NRM = PCA_CCR_NRM[self.ell]
        self.PCA_Ev = PCA_Ev
        self.PCA_CCR = PCA_CCR
        self.n_pca = n_pca
        self.PCA_Ev_NRM = PCA_Ev_NRM
        self.L = np.diag(np.sqrt(PCA_Ev_NRM[: self.ell_max] / PCA_Ev[: self.ell_max]))
        self.X_mean = np.mean(X, axis=0)
        self.PCA_Ev_sum_all = PCA_Ev_sum_all
        self.noise_var = noise_var
        self.fit_idx = True

    def transform(self, X, return_ess=False):
        """
        Apply RECODE to X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features).
                Transsforming data matrix, where `n_samples` is the number of samples
                and `n_features` is the number of features.
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
                Denoised data matrix.
        """
        if self.fit_idx == False:
            raise TypeError("Run fit before transform.")
        if self.method == "variance":
            return self._noise_reduct_noise_var(X, self.noise_var)
        elif self.method == "manual":
            self.ell = self.ell_manual
            return self._noise_reductor(
                X, self.L, self.U, self.X_mean, self.ell, self.version, return_ess
            )

        self.RECODE_done = True

    def fit_transform(self, X):
        """
        Fit and transform RECODE to X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features).
                Transsforming data matrix, where `n_samples` is the number of samples
                and `n_features` is the number of features.
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
                Denoised data matrix.
        """
        self.fit(X)
        return self.transform(X)
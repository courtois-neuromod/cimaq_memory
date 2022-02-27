
# def make_fit_1stlevel_glm(fmri_img:Union[str,os.PathLike,Nifti1Image],
#                           events:pd.DataFrame,
#                           confounds:pd.DataFrame=None,
#                           **kwargs):
#     short_events = events[['onset','trial_type','duration']].copy(deep=True)
# #     Add modulation column to events since ctl classification is 2x poorer than enc?
# #     short_events['modulation'] = [0.2 if 'Enc' in ttype else 0.9
# #                                   for ttype in short_events.trial_type]


#     glm_params = dict(t_r=fmri_img.header.get_zooms()[-1],
#                       mask_img=mask_img,
#                       drift_model=design_params['drift_model'],
#                       standardize=False,
#                       smoothing_fwhm=None,
#                       signal_scaling=False,
#                       noise_model='ar1',
#                       hrf_model=design_params['hrf_model'],
#                       minimize_memory=False)
#     design = make_first_level_design_matrix(**design_params)
#     model = FirstLevelModel(**glm_params).fit(fmri_img, design_matrices=design)
#     return model

#############################
# def trial_fmri(fmri_path:Union[str,os.PathLike, Nifti1Image],
#                events_path:Union[str,os.PathLike, pd.DataFrame],
#                sep:str='\t',
#                **kwargs):
#     from itertools import starmap
#     from nilearn import image as nimage
#     import pandas as pd
#     # Make pandas Intervals (b:list of beginnigs, e:list of ends)
#     mkintrvls = lambda b, e: list(starmap(pd.Interval,tuple(zip(b, e))))
#     fmri_img = nimage.load_img(fmri_path)
#     events = [events if isinstance(events, pd.DataFrame)
#                else pd.read_csv(events_path, sep=sep)][0]
#     t_r = fmri_img.header.get_zooms()[-1]
#     frame_times = np.arange(fmri_img.shape[-1]) * t_r
#     frame_intervals = mkintrvls(pd.Series(frame_times).values,
#                                 pd.Series(frame_times).add(t_r).values)
#     trial_ends=(events.onset+abs(events.onset -
#                                  events.offset)+events.isi).values
#     trial_intervals = mkintrvls(events.onset.values, trial_ends)
# #     trial_intervals = list(starmap(pd.Interval,tuple(zip(events.onset.values, trial_ends))))
#     bold_by_trial_indx = [[frame[0] for frame in enumerate(frame_intervals)
#                            if frame[1].left in trial] for trial in trial_intervals]
#     bold_by_trial = list(nimage.index_img(fmri_img, idx)
#                          for idx in bold_by_trial_indx)
#     event_list = events.loc[[item[0] for item in
#                               enumerate(bold_by_trial) if item != []]]
#     mem_labels, recall_labels = events_list.contidion, events_list.recognition_performance
#     return bold_by_trial, mem_labels, recall_labels
###############################

# def trial_fmri(fmri_path:Union[str,os.PathLike, Nifti1Image],
#                events_path:Union[str,os.PathLike, pd.DataFrame],
#                sep:str='\t', t_r:float=None,
#                **kwargs):
#     from itertools import starmap
#     from more_itertools import flatten
#     from nilearn import image as nimage
#     import pandas as pd
#     # Make pandas Intervals (b:list of beginnigs, e:list of ends)
#     mkintrvls = lambda b, e: list(starmap(pd.Interval,tuple(zip(b, e))))
#     fmri_img = nimage.load_img(fmri_path)
#     if not isinstance(events_path, pd.DataFrame):
#         events = pd.read_csv(events_path, sep=sep)
#     else:
#         events = events_path
#     t_r = [t_r if t_r is not None else
#            fmri_img.header.get_zooms()[-1]][0]
#     frame_times = np.arange(fmri_img.shape[-1]) * t_r
#     frame_ends = pd.Series(frame_times).add(t_r).values
#     frame_intervals = mkintrvls(pd.Series(frame_times).values,
#                                 frame_ends)
#     trial_ends=(events.onset+abs(events.onset -
#                                  events.offset)+events.isi).values
#     trial_intervals = mkintrvls(events.onset.values, trial_ends)

#     valid_trial_idx = [trial[0] for trial in enumerate(trial_intervals)
#                        if trial[1].left<frame_intervals[-1].left]
#     valid_trials = pd.Series(trial_intervals).loc[valid_trial_idx].values
# #     trial_intervals = list(starmap(pd.Interval,tuple(zip(events.onset.values, trial_ends))))
#     bold_by_trial_indx = [[frame[0] for frame in enumerate(frame_intervals)
#                            if frame[1].left in trial] for trial in valid_trials]
# #     bold_by_trial = list(nimage.index_img(fmri_img, idx)
# #                          for idx in bold_by_trial_indx)
#     valid_frame_intervals = [pd.Series(frame_intervals).loc[bold_idx].values
#                              for bold_idx in bold_by_trial_indx]
#     perfo_labels = events.iloc[valid_trial_idx].recognition_performance.fillna('Ctl')
#     condition_labels = events.iloc[valid_trial_idx].trial_type
#     stim_labels = events.iloc[valid_trial_idx].stim_file.fillna('Ctl').values
#     categ_labels = events.iloc[valid_trial_idx].stim_category.fillna('Ctl').values
    
#     return pd.DataFrame(tuple(zip(valid_trial_idx,
# #                                   bold_by_trial,
#                                   bold_by_trial_indx,
#                                   valid_trials,
#                                   valid_frame_intervals,
#                                   condition_labels,
#                                   perfo_labels, stim_labels, categ_labels)),
#                         columns=['trials',
# #                                  'trial_niftis',
#                                  'fmri_frames',
#                                  'trial_intervals', 'fmri_frame_intervals',
#                                  'condition_labels', 'performance_labels',
#                                  'stimuli_files', 'category_labels'])

# def make_fit_1stlevel_glm(fmri_img:Union[str,os.PathLike,Nifti1Image],
#                           events:pd.DataFrame,
#                           confounds:pd.DataFrame=None,
#                           **kwargs):
#     short_events = events[['onset','trial_type','duration']].copy(deep=True)
#     Add modulation column to events since ctl classification is 2x poorer than enc?
#     short_events['modulation'] = [0.2 if 'Enc' in ttype else 0.9
#                                   for ttype in short_events.trial_type]
#     sess00.condition_events = sess00.events[['onset','trial_type','duration']]
#     sess00.performance_events = sess00.events.rename({'recognition_performance':'trial_type'},
#                                              axis=1)[['onset','trial_type','duration']]
#     design_params = dict(frame_times=(np.arange(sess00.cleaned_fmri.shape[-1]) *
#                                       fmri_img.header.get_zooms()[-1]),
#                          events=sess00.condition_events[['onset','trial_type','duration']],
#                          drift_model=None,
#                          hrf_model='spm')

# glm_params = dict(t_r=sess00.cleaned_fmri.header.get_zooms()[-1],
#                   mask_img=sess00.full_mask_img,
#                   drift_model=design_params['drift_model'],
#                   standardize=False,
#                   smoothing_fwhm=None,
#                   signal_scaling=(0,1), # Other choices: 0, 1, False
#                   noise_model='ar1',
#                   hrf_model=design_params['hrf_model'],
#                   subject_label=sess00.sub_id,
#                   target_affine=sess00.cleaned_fmri.affine,
#                   target_shape=sess00.full_mask_img.shape,
#                   n_jobs=-1,
#                   minimize_memory=False)
    # Signal extraction methods: 
    # compute_contrast(self, contrast_def, stat_type=None, output_type='z_score')
    # DONE fit(self, run_imgs, events=None, confounds=None, design_matrices=None, bins=100)
    # predicted(), r_square(), residuals
    # fit_transform(self, X, y=None, **fit_params)
#     design = make_first_level_design_matrix(**design_params)
#     model = FirstLevelModel(**glm_params).fit(sess00.cleaned_fmri,
#                                               design_matrices=design)
    
#     return model


# def weightings(signals: pd.DataFrame,
#                weights: pd.DataFrame,
#                condition_labels: Union[list, pd.Index]
#                ) -> pd.DataFrame:
#     """
#     Return condition-wise weighted signals DataFrame.
#     """

#     newsignals = signals.set_index(condition_labels)
#     for cond in weights.index:
#         newsignals.loc[cond] = newsignals.loc[cond]*weights.loc[cond]
#     return newsignals


# def get_weighted_signals(signals: pd.DataFrame,
#                          weights: list,
#                          condition_labels: Union[list, pd.Index],
#                          standardize: bool = True
#                          ) -> pd.DataFrame:
#     """
#     Return condition-wise weighted signals DataFrame for all conditions.
    
#     Args:
#         signals
#     """

#     orig_index, orig_cols = signals.index, signals.columns
#     w0 = [weightings(signals, weights=weights[cond],
#                      condition_labels=cond_labels[cond])
#           for cond in range(len(weights))]
#     slist = [signals]+w0
#     data = np.prod(np.array(slist), axis=0)
#     if standardize is True:
#         data = StandardScaler().fit_transform(data)
#     return pd.DataFrame(data, index=orig_index, columns=orig_cols)


# def manage_duplicates(X, method='mean', axis=0):
#     """
#     Returns ``X`` without duplicate labels along ``axis``.
#     """

#     from inspect import getmembers
#     dups = X.loc[:,X.columns.value_counts()>1].columns.unique()
#     if len(dups) == 0:
#         return X
#     mthd = dict(getmembers(pd.DataFrame))[f'{method}']
#     newdata = [pd.Series(data=X[dup].T.apply(mthd), name=dup)
#                for dup in dups]
#     return pd.concat([X.copy(deep=True).drop(dups, axis=1),
#                       pd.concat(newdata, axis=1)], axis=1)


# def weight_signals(signals, weights, labels,
#                    labelize=True):
#     """
#     Returns the product of the contrasts obtained with ``get_glm_events``.
    
#     Signals, weights and labels should be aligned along a common axis.
#     Otherwise, ValueError is raised.
#     """
    
#     signals = signals.set_axis(labels, axis=0)
#     for weight in weights.index:
#         signals.loc[weight] = signals.loc[weight] * weights.loc[weight]
#     if labelize is True:
#         signals = signals.set_index(labels)
#     return signals, labels


# def weight_signals(signals, weights, labels, keep_zero_var=False):
#     """
#     Returns the product of the contrasts obtained with ``get_glm_events``.
#     """

#     from sklearn.feature_selection import VarianceThreshold
#     signals = signals.set_index(labels)
#     for weight in weights.index:
#         signals.loc[weight] = signals.loc[weight]*weights.loc[weight]
#     if keep_zero_var is False:
#         cols = VarianceThreshold().fit(signals).get_support(indices=True)
#         signals = signals.iloc[:, cols]
#     return signals, labels


#  def weight_signals(signals, weights, labels,
#                    labelize=True):
#     """
#     Returns the product of the contrasts obtained with ``get_glm_events``.
    
#     Signals, weights and labels should be aligned along a common axis.
#     Otherwise, ValueError is raised.
#     """
    
#     signals = signals.set_axis(labels, axis=0)
#     for weight in weights.index:
#         signals.loc[weight] = signals.loc[weight] * weights.loc[weight]
#     if labelize is True:
#         signals = signals.set_index(labels)
#     return signals, labels


# def get_contrasts(fmri_img=None,
#                   events=None,
#                   design_kws: Union[dict, Bunch] = None,
#                   glm_kws: Union[dict, Bunch] = None,
#                   trial_type_cols: list = None,
#                   method='mean',
#                   output_type: str = 'effect_variance',
#                   masker: [MultiNiftiMasker, NiftiLabelsMasker,
#                            NiftiMapsMasker, NiftiMasker] = None,
#                   labels: Sequence = None,
#                   session=None,
#                      **kwargs):

#     design_defs, glm_defs = {}, {}

#     if session is not None:
#         fmri_img, events = itemgetter(*['cleaned_fmri',
#                                         'events'])(session)
#         design_defs.update(session.design_defs)
#         glm_defs.update(session.glm_defs)

#     t_r = get_t_r(fmri_img)
#     frame_times = get_frame_times(fmri_img)

#     if design_kws is not None:
#         design_defs.update(design_kws)
#     if glm_kws is not None:
#         glm_defs.update(glm_kws)

#     design = make_first_level_design_matrix(frame_times, events=events,
#                                             drift_model=None,
#                                             **design_defs)

#     model = FirstLevelModel(**glm_defs).fit(run_imgs=fmri_img,
#                                             design_matrices=design)
#     contrasts = nimage.concat_imgs([model.compute_contrast(trial,
#                                                            output_type=output_type)
#                                     for trial in tqdm(design.columns.astype(str),
#                                                       desc='Computing Contrasts')])
#     if masker is None:
#         masker = NiftiMasker().fit(contrasts)

#     signals = masker.transform_single_imgs(contrasts)
#     signals = pd.DataFrame(signals, columns=labels,
#                            index=design.columns).iloc[:-1, :]
#     return Bunch(model=model, contrast_img=contrasts, signals=signals)


# def preprocess_events(events: Union[str, PathLike,
#                                     PosixPath, pd.DataFrame],
#                       fmri_img: Union[str, PathLike,
#                                       PosixPath, Nifti1Image]):
#     from nilearn.image import load_image
    
#     fmri_img = load_img(fmri_img)

#     if not isinstance(pd.DataFrame, events):
#         events = pd.read_csv(StringIO(Path(events).read_text().lower()),
#                              sep='\t').drop('trial_number',
#                                             axis=1).reset_index(
#                      drop=False).rename(
#                          {'index': 'trial_number'},
#                          axis=1).set_index('trial_number')
#     events['duration'] = events['duration'] + events['isi']
#     events['trial_ends'] = events.onset + events.duration
#     events = events[~(events.trial_ends > fmri_img.shape[-1]*)] 
#     events.stim_file = events.stim_file.fillna('empty_box_gris.bmp')
#     events.stim_category = events.stim_category.fillna('ctl')
#     events.stim_id = events.stim_id.fillna('ctl')
#     events.recognition_performance.fillna('ctl', inplace=True)
#     events['position_performance'] = ['cs' if 'hit' ==
#                                       row[1].recognition_performance
#                                       else 'ws' if
#                                       row[1].recognition_accuracy
#                                       is True and
#                                       row[1].position_accuracy is False
#                                       else
#                                       row[1].recognition_performance
#                                       for row in events.iterrows()]
#     return events


# def get_optimal_features_recursive(estimator,
#                                    X: Iterable,
#                                    y: Iterable,
#                                    cv: Union[int, callable] = None,
#                                    step: Union[int, float] = 1,
#                                    min_features_to_select: int = 1,
#                                    n_jobs: int = 1,
#                                    scoring: Union[str, callable] = 'accuracy',
# #                                    cv_kws: Union[dict, Bunch] = None,
#                                    **kwargs
#                                    ) -> Iterable:
#     from builtins import FutureWarning
#     import warnings
#     from sklearn.model_selection import StratifiedKFold
#     from sklearn.feature_selection import SelectKBest, SelectFwe
#     from sklearn.feature_selection import SelectFdr, SelectFpr
#     from sklearn.feature_selection import RFE, RFECV
#     if not isinstance(X, pd.DataFrame):
#         X = pd.DataFrame(X)

#     est_params = dict(estimator=estimator,
#                       step=step,
#                       cv=cv, n_jobs=n_jobs,
#                       importance_getter='auto',
#                       min_features_to_select=min_features_to_select,
#                       scoring=scoring)
    
#     refcv = RFECV(**est_params)
#     if isinstance(y, (list, tuple)):
#         return [refcv.fit_transform(X.copy(deep=True), task[1])
#                       for task in enumerate(y)]
#     else:
#         rfecv.fit(X.copy(deep=True), task[1])
#         return X.iloc[:, rfecv.get_support(indices=True)]
    

# def get_masker(src: Union[str, PathLike, PosixPath],
#                sub_id: str = None,
#                ses_id: str = None,
#                task: str = None,
#                space: str = None,
#                session: Union[dict, Bunch] = None,
#                encoding: str = None,
#                **kwargs
#                ) -> [MultiNiftiMasker, NiftiLabelsMasker,
#                      NiftiMapsMasker, NiftiMasker]:
#     import pickle
#     import sys
#     from operator import itemgetter

#     if session is not None:
#         attrs = ['sub_id', 'ses_id', 'task', 'space', 'masker_path']
#         sub_id, ses_id, task, space, masker_path = itemgetter(*attrs)(session)
#         src = masker_path
#     masker_str = f'*{sub_id}*wholebrain*.pickle'
#     if encoding is None:
#         encoding = sys.getdefaultencoding()
#     if src is None:
#         return None
#     masker_path = list(Path(src).rglob(masker_str))[0]
#     if masker_path == []:
#         return None
#     with open(masker_path, mode='rb') as mfile:
#         masker = pickle.load(mfile, encoding=encoding)
#         mfile.close()
#     return masker


# def get_weighted_signals(signals: pd.DataFrame,
#                          weights: list,
#                          #                          standardize: bool = True,
#                          **kwargs
#                          ) -> pd.DataFrame:
#     """
#     Return condition-wise weighted signals DataFrame for all conditions.

#     Args:
#         signals
#     """

#     orig_index, orig_cols = signals.index, signals.columns

#     w0 = [weightings(signals, weights=weights[weight[0]])
#           for weight in enumerate(weights)]
#     slist = [signals]+w0
#     data = np.sum(np.array(slist), axis=0)
# #     if standardize is True:
# #         data = StandardScaler().fit_transform(data)
#     return pd.DataFrame(data, index=orig_index, columns=orig_cols)


# def SortFeaturesByConditionPCA(X: Iterable, y: Iterable,
#                                method: str = 'pearson',
#                                n_features: Union[int, float] = None
#                                ) -> Bunch:

#     pairwise_metrics = get_pairwise_metrics()

        
#     if not isinstance(y, pd.Series):
#         y = pd.Series(y)
#     if not isinstance(X, pd.DataFrame):
#         X = pd.DataFrame(X)
#     if isinstance(n_features, float):
#         n_features = int(round(X.shape[1] * n_features, 0))
#     X = X.set_axis(y,axis=0)
#     classes_ = y.unique()
    
#     estimators_ = []
#     for cond in y.unique():
#         try:
#             if method in pairwise_metrics.index:
#                 method = pairwise_metrics.loc[method]
#                 estimators_.append(PCA().fit(pd.DataFrame(method(X.loc[cond].T),
#                                                           index=X00.columns,
#                                                           columns=X00.columns)))
#             else:
#                 estimators_.append(PCA().fit(X.loc[cond].corr(method)))
#         except (TypeError, ValueError):
#             continue

#     feature_names_out_ = [np.array([[name[1] for name in
#                                      enumerate(pca.feature_names_in_)
#                                     if name[0]==idx][0] for idx in
#                                    np.argsort(
#                                        pca.explained_variance_
#                                               )][:n_features])
#                           for pca in estimators_]
#     names_ = Bunch(**dict(tuple(zip(classes_, feature_names_out_))))
#     names_.explained_variance_ratios_ = [est.explained_variance_ratio_[:n_features]
#                                          for est in estimators_]
#     return names_


# def get_pairwise_metrics():
#     import inspect
#     import scipy
#     import sklearn
#     from operator import itemgetter

#     sklearn_func_names = ['cityblock', 'cosine', 'euclidean',
#                           'l1', 'l2', 'manhattan']
#     sklearn_dict = sklearn.metrics.pairwise.distance_metrics()
#     sklearn_funcs = pd.Series(itemgetter(*sklearn_func_names)(sklearn_dict),
#                                  index=sklearn_func_names)

#     scipy_func_names = ['braycurtis','canberra', 'chebyshev',
#                         'correlation', 'dice', 'hamming', 'jaccard',
#                         'kulsinski', 'mahalanobis', 'minkowski',
#                         'rogerstanimoto', 'russellrao', 'seuclidean',
#                         'sokalmichener', 'sokalsneath',
#                         'sqeuclidean', 'yule']
#     scipy_dict = dict(inspect.getmembers(scipy.spatial.distance))
#     scipy_funcs = pd.Series(itemgetter(*scipy_func_names)(scipy_dict),
#                             index=scipy_func_names)

#     callables_ = pd.concat([sklearn_funcs, scipy_funcs])
#     return callables_


# def get_optimal_features_recursive(estimator,
#                                    X: Iterable, y: Iterable,
#                                    cv: Union[int, callable] = StratifiedKFold,
#                                    step: Union[int, float] = 1,
#                                    min_features_to_select: int = 1,
#                                    n_jobs: int = 1,
#                                    scoring: Union[str, callable] = 'accuracy',
#                                    **kwargs
#                                    ) -> Iterable:
#     """
#     Recursively drop ``step`` feature at each iteration using cross-validation.
#     """

#     from sklearn.feature_selection import RFECV
#     # from builtins import FutureWarning
#     # import warnings
#     # warnings.filterwarnings(action='ignore', category=FutureWarning,
#     #                         module='sklearn.utils')
#     if not isinstance(X, pd.DataFrame):
#         X = pd.DataFrame(X)
#     if not isinstance(cv, int):
#         cv = cv()
#     rec = RFECV(estimator, step=step,
#                 cv=cv, n_jobs=n_jobs,
#                 min_features_to_select=min_features_to_select,
#                 scoring=scoring)

#     return X.iloc[:, rec.fit(X, y).get_support(indices=True)]


# def get_optimal_features(X: [np.ndarray, pd.DataFrame]
#                          ) -> list:
#     """
#     Return optimal features using hierarchical clustering.


#     """

#     from scipy.cluster import hierarchy

#     if isinstance(X, pd.DataFrame):
#         corr = X.corr(method='spearman').fillna(0).values
#         np.fill_diagonal(corr, 1)
#     else:
#         corr = spearmanr(X).correlation
#         # Ensure the correlation matrix is symmetric
#         corr = (corr + corr.T) / 2
#         np.fill_diagonal(corr, 1)
#         np.nan_to_num(corr, 0)
#     # Converting the correlation matrix to a distance matrix
#     distance_matrix = 1 - np.abs(corr)
#     # hierarchical clustering using Ward's linkage
#     dist_linkage = hierarchy.ward(squareform(distance_matrix))

#     cluster_ids = hierarchy.fcluster(dist_linkage, 1, criterion="distance")
#     cluster_id_to_feature_ids = defaultdict(list)

#     for idx, cluster_id in enumerate(cluster_ids):
#         cluster_id_to_feature_ids[cluster_id].append(idx)
#     selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
#     return selected_features


# class FMRIPrepPathMatcher(sklearn.utils.Bunch):
#     from glob import glob
#     from cimaq_decoding_pipeline import fetch_fmriprep_session
    
#     @classmethod
#     def fetch(self, **kwargs):
        
#         return fetch_fmriprep_session(session=vars(self))

#     def __init__(self,
#                  fmri_path,
#                  anat_modality: str = 'T1w',
#                  **kwargs):
#         import re
#         from os.path import basename, dirname, splitext
        
#         from cimaq_decoding_utils import get_events

#         self.fmri_path = fmri_path
#         if kwargs is not None:
#             [setattr(self, item[0], item[1])
#              for item in tuple(kwargs.items())]
#         if hasattr(self, 'events_dir'):
#             self.events_path = get_events(self.fmri_path,
#                                           self.events_dir)            
#         self.anat_modality = anat_modality
#         cutoff = re.search('(fmriprep/)',
#                            fmri_path).span()[1]
#         fmriprep_dir = Path(fmri_path[:cutoff])
#         ext = fmri_path.split('.', maxsplit=1)[1]
#         bids_parts = ['sub_id', 'ses_id', 'task',
#                       'space', 'desc', 'modality']
#         bids_values = basename(fmri_path).split('_')
#         sub_id, ses_id, task, space, desc, modality = bids_values

#         anat_suffix = '_'.join([sub_id, f'*{space}_{desc}',
#                                 f'{anat_modality}.{ext}'])
#         mask_suffix = '_'.join([sub_id, ses_id, task, space,
#                                 f'desc-brain_mask.{ext}'])
#         events_suffix = '_'.join([sub_id, ses_id,
#                                   task, 'events.tsv'])
#         behav_suffix = '_'.join([sub_id, ses_id, task,
#                                  'behavioural.tsv'])      
#         self.anat_path = sorted(fmriprep_dir.rglob(anat_suffix))[0]
        
#         self.mask_path = sorted(Path(fmriprep_dir).rglob(mask_suffix))[0]
        
# #         authorized = ['events_dir', 'behavioural_dir',
# #                       'behav_dir', 'masker_dir']

#         [setattr(self, itm[0], itm[1]) for itm
#          in tuple(zip(bids_parts, bids_values))]
    
#         setattr(self, '__dict__', vars(self))
#         setattr(self, 'fetch', (fetch_fmriprep_session(session=vars(self))))


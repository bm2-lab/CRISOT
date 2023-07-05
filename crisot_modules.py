import pandas as pd
import numpy as np
import os
import pickle

pwd = os.path.dirname(os.path.realpath(__file__))

def a_b_cal(param_read):
    max_val = param_read.values.max(axis=0)
    min_val = param_read.values.min(axis=0)
    min_match = param_read.loc[['AA', 'CC', 'GG', 'TT'], :].values.min(axis=0)
    max_sc = max_val.sum()
    min_sc = np.append(np.sort(min_match)[:-6], np.sort(min_val)[:6]).sum()
    a = 1 / (max_sc - min_sc)
    b = -(min_sc / (max_sc - min_sc))
    return [a, b]

class CRISOT:
    def __init__(self, param, ref_genome, a_b=None, cutoff=None, opti_th=None, prob_weight='default', bins=None):
        self.feat_dict = {}
        for key in param.index:
            for i in range(20):
                self.feat_dict['Pos' + str(i + 1) + '_' + key] = param.loc[key, :].values[i]
        self.ref_genome = ref_genome
        if a_b == None:
            a_b = a_b_cal(param)
        self.a_b = a_b
        if cutoff == None:
            cutoff = 0.
        self.cutoff = cutoff
        if opti_th == None:
            opti_th = 0.8
        self.opti_th = opti_th
        if prob_weight == 'default':
            prob_weight = np.array([0, 0.000007, 0.000025, 0.000413, 0.006692, 0.056202, 0.239645, 0.652542, 1.0])
        self.prob_weight = prob_weight
        if bins == None:
            bins = [0, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0]
        self.bins = bins

    def single_score_(self, on_seq, off_seq):
        y_pred = np.array([self.feat_dict['Pos' + str(j + 1) + '_' + on_seq[j] + off_seq[j]] for j in range(20)]).sum()
        y_pred = y_pred * self.a_b[0] + self.a_b[1]
        return y_pred

    def score(self, data_path=None, data_df=None, On='On', Off='Off', Active=None):
        if data_df is not None:
            data_set = data_df
        else:
            data_set = pd.read_csv(data_path, sep=",", header=0, index_col=None)
        ont = data_set.loc[:, On].values
        offt = data_set.loc[:, Off].values
        y_pred = np.array([self.single_score_(ont[i], offt[i]) for i in range(ont.shape[0])])
        if Active == None:
            return y_pred
        else:
            y_ori = data_set.loc[:, Active].values
            return y_ori, y_pred

    def score_bin_(self, y_pred):
        y_pred = np.array(y_pred)
        if y_pred.shape[0] != 0:
            y_df = pd.DataFrame(y_pred.reshape(-1,1), columns=['CRISOT-Score'])
            y_count = y_df['CRISOT-Score'].value_counts(bins=self.bins, sort=False).values
        else:
            y_count = np.array([0] * (len(self.bins)-1))
        return y_count

    def single_aggre_(self, y_pred, out_cnt=True):

        cnt = self.score_bin_(y_pred)
        aggre = (cnt * self.prob_weight).sum()
        if out_cnt:
            return np.append(cnt[-4:], aggre)
        else:
            return aggre

    def aggre(self, data_path=None, data_df=None, On='On', Off='Off', target=None, out_df=False, out_cnt=True):
        if data_df is not None:
            data_set = data_df
        else:
            data_set = pd.read_csv(data_path, sep=",", header=0, index_col=None)
        offt = data_set.loc[:, Off].values
        if target is not None:
            assert len(target) == 23, 'target sequence must have 23 nt'
            assert data_set.loc[offt == target, :].shape[0] > 0, 'No sequence match the target sequence'
            data_set = pd.concat([data_set.loc[offt == target, :], data_set.loc[offt != target, :]])
        y_pred = self.score(data_df=data_set, On=On, Off=Off, Active=None)
        aggre = self.single_aggre_(y_pred[1:], out_cnt)
        if out_df:
            data_set['CRISOT-Score'] = y_pred
            return aggre, data_set
        else:
            return aggre

    def single_spec_(self, y_pred):
        aggre = self.single_aggre_(y_pred, out_cnt=False)
        spec = 10 / (10 + aggre)
        return spec

    def spec(self, data_path=None, data_df=None, On='On', Off='Off', target=None, out_df=False):
        if data_df is not None:
            data_set = data_df
        else:
            data_set = pd.read_csv(data_path, sep=",", header=0, index_col=None)
        offt = data_set.loc[:, Off].values
        if target is not None:
            assert len(target) == 23, 'target sequence must have 23 nt'
            assert data_set.loc[offt == target, :].shape[0] > 0, 'No sequence match the target sequence'
            data_set = pd.concat([data_set.loc[offt == target, :], data_set.loc[offt != target, :]])
        y_pred = self.score(data_df=data_set, On=On, Off=Off, Active=None)
        spec = self.single_spec_(y_pred[1:])
        if out_df:
            data_set['CRISOT-Score'] = y_pred
            return spec, data_set
        else:
            return spec

    def CasoffinderSpec_(self, sgrna, target, out_df=False, offtar_search=None, mm=6, dev='C'):
        if offtar_search is not None:
            offtar_search = offtar_search
        else:
            offtar_search = os.path.join(pwd, 'script/casoffinder_genome.sh')
        if os.path.exists('.temp_casoffinder.out'):
            os.system('rm .temp_casoffinder.out')
        if os.path.exists('.temp_casoffinder.in'):
            os.system('rm .temp_casoffinder.in')
        os.system("sh {} {} {} {} {}".format(offtar_search, sgrna[:20], self.ref_genome, mm, dev))
        data_set = pd.read_csv('.temp_casoffinder.out', sep="\t", header=None, index_col=None)
        offt = data_set.loc[:, 3].values
        offt = np.array([str.upper(t) for t in offt])
        data_set.loc[:, 3] = offt
        data_set = data_set[-data_set[3].str.contains('N|R|W|M|V|Y|K|D|S|J')]
        data_set.drop_duplicates([1,2,3], inplace=True)
        if data_set[data_set[3].str.contains(target[:20])].shape[0] == 0:
            data_set = data_set.append(pd.Series([data_set.iloc[0, 0], np.nan, np.nan, target, np.nan, np.nan]), ignore_index=True)
        data_set = pd.concat(
            [data_set[data_set[3].str.contains(target[:20])], data_set[-data_set[3].str.contains(target[:20])]])
        if out_df:
            spec, out_dset = self.spec(data_df=data_set, On=0, Off=3, out_df=out_df)
        else:
            spec = self.spec(data_df=data_set, On=0, Off=3)
        os.system("rm .temp_casoffinder.out")
        if out_df:
            return spec, out_dset
        else:
            return spec

    def opti(self, target, opti_type=None, ref=None, offtar_search=None, accepted_mutate=None, cd33cut=0.6, mm=6, dev='G0'):
        # opti_type: how to mutate the sgRNA? default is 3 types of mutations for each position
        if opti_type == None:
            opti_pos = []
            opti_seq = []
            for p in range(1, 21):
                opti_pos = np.append(opti_pos, np.array([p, p, p, p], dtype='int'))
                opti_seq = np.append(opti_seq, ['A', 'C', 'G', 'T'])
            opti_nt = pd.DataFrame()
            opti_nt['Pos'] = opti_pos
            opti_nt['nt'] = opti_seq
        elif type(opti_type) == tuple:
            opti_pos = []
            opti_seq = []
            for p in range(opti_type[0], opti_type[1]):
                opti_pos = np.append(opti_pos, np.array([p, p, p, p], dtype='int'))
                opti_seq = np.append(opti_seq, ['A', 'C', 'G', 'T'])
            opti_nt = pd.DataFrame()
            opti_nt['Pos'] = opti_pos
            opti_nt['nt'] = opti_seq
        elif (type(opti_type) == np.ndarray) or (type(opti_type) == list) or (type(opti_type) == range):
            opti_pos = []
            opti_seq = []
            for p in opti_type:
                opti_pos = np.append(opti_pos, np.array([p, p, p, p], dtype='int'))
                opti_seq = np.append(opti_seq, ['A', 'C', 'G', 'T'])
            opti_nt = pd.DataFrame()
            opti_nt['Pos'] = opti_pos
            opti_nt['nt'] = opti_seq
        elif type(opti_type) == pd.core.frame.DataFrame:
            opti_nt = opti_type
        else:
            opti_nt = pd.read_csv(opti_type, header=0, index_col=0)

        mut = [None]
        if accepted_mutate is not None:
            if type(accepted_mutate) == pd.core.frame.DataFrame:
                accepted_list = accepted_mutate.loc[accepted_mutate['Percent-Active']>=cd33cut, ['Position', 'Mismatch']].values
                for i in range(accepted_list.shape[0]):
                    mut.append(str(accepted_list[i,0]) + accepted_list[i,1])
        accepted_list = mut

        results = []
        # calculate WT spec score
        spec, out_df = self.CasoffinderSpec_(target, target, out_df=True, offtar_search=offtar_search, mm=mm, dev=dev)
        cnt_results = self.score_bin_(out_df.loc[:, 'CRISOT-Score'].values[1:])[-4:]
        if ref is not None:
            ref_merge = ref.loc[:, ['Offtarget_Sequence', 'GUIDE-Seq Reads']]
            ref_merge.columns = [3, 'Reads']
            out_merge = pd.merge(out_df.iloc[1:,:], ref_merge, how='left', on=[3], sort=False)
            out_merge.fillna(0, inplace=True)
            out_merge = out_merge.loc[out_merge.loc[:,'Reads'] > 0, :]
            if out_merge.shape[0] != 0:
                out_merge.to_csv('{}_ref_match.tsv'.format(target), sep='\t')
                spec_ref = self.single_spec_(out_merge.loc[:, 'CRISOT-Score'].values[1:], out_cnt=False)
            else:
                spec_ref = 1.0
            results.append([target, target, 'WT', out_df.loc[:, 'CRISOT-Score'].values[0]] + list(cnt_results) + [spec, spec_ref])
        else:
            results.append([target, target, 'WT', out_df.loc[:, 'CRISOT-Score'].values[0]] + list(cnt_results) + [spec])
        # calculate spec scores of modified sgRNAs
        for i in range(opti_nt.shape[0]):
            pos = int(opti_nt.loc[i, 'Pos'])
            nt = opti_nt.loc[i, 'nt']
            if target[pos - 1] != nt:
                new_sgrna = target[:pos - 1] + nt + target[pos:]
                if accepted_mutate is not None:
                    mut_nt = str(pos) + nt + target[pos-1]
                else:
                    mut_nt = None
                if (self.single_score_(new_sgrna, target) > self.opti_th) and (mut_nt in accepted_list):
                
                    spec, out_df = self.CasoffinderSpec_(new_sgrna, target, out_df=True, offtar_search=offtar_search, mm=mm, dev=dev)
                    cnt_results = self.score_bin_(out_df.loc[:, 'CRISOT-Score'].values[1:])[-4:]
                    if ref is not None:
                        out_merge = pd.merge(out_df.iloc[1:,:], ref_merge, how='left', on=[3], sort=False)
                        out_merge.fillna(0, inplace=True)
                        out_merge = out_merge.loc[out_merge.loc[:,'Reads'] > 0, :]
                        if out_merge.shape[0] != 0:
                            spec_ref = self.single_spec_(out_merge.loc[:, 'CRISOT-Score'].values[1:], out_cnt=False)
                        else:
                            spec_ref = 1.0
                    if ref is not None:
                        results.append([new_sgrna, target, '{}{}>{}'.format(target[pos - 1], pos, nt), out_df.loc[:, 'CRISOT-Score'].values[0]] + list(cnt_results) + [spec, spec_ref])
                    else:
                        results.append([new_sgrna, target, '{}{}>{}'.format(target[pos - 1], pos, nt), out_df.loc[:, 'CRISOT-Score'].values[0]] + list(cnt_results) + [spec])

        results = np.array(results)
        if results.shape[0] == 1:
            results = np.vstack([results, np.array([[np.nan for i in range(results.shape[1])]], dtype=object)])
            results[1, 1:3] = np.array([target, 'Optimization unavailable'])
        
        if ref is not None:
            results_df = pd.DataFrame(results, columns=['sgRNA', 'Target', 'Mutation', 'CRISOT-Score', '(0.65, 0.7]', '(0.7, 0.75]', 
                                                        '(0.75, 0.8]', '(0.8, 1.0]', 'CRISOT-Spec', 'ref_Spec'])
        else:
            results_df = pd.DataFrame(results, columns=['sgRNA', 'Target', 'Mutation', 'CRISOT-Score', '(0.65, 0.7]', '(0.7, 0.75]', 
                                                        '(0.75, 0.8]', '(0.8, 1.0]', 'CRISOT-Spec'])
        results_df.iloc[:, 3:] = np.array(results_df.iloc[:, 3:].values, dtype='float')
        try:
            results_df.iloc[:, 4:8] = np.array(results_df.iloc[:, 4:8].values, dtype='int')
        except:
            results_df.iloc[0, 4:8] = np.array(results_df.iloc[0, 4:8].values, dtype='int')
        results_df['delta_Spec'] = results_df.loc[:, 'CRISOT-Spec'].values - results_df.loc[0, 'CRISOT-Spec']
        results_df['rk'] = np.array([0] + [1] * (results_df.shape[0]-1))

        results_out = results_df.sort_values(by=['rk', 'delta_Spec'], ascending=[True, False])
        
        results_out.index = np.arange(results_out.shape[0])
        return results_out.iloc[:, :-1]

    def opti_chopchop(self, chop_tsv=None, chop_df=None, top=10, opti_type=None, offtar_search=None, mm=6, dev='G0', 
                      accepted_mutate=None, cd33cut=0.6):
        if chop_tsv is not None:
            if len(chop_tsv) > 4:
                if chop_tsv[-4:] == '.tsv':
                    chop_tsv = chop_tsv[:-4]
            dataread = pd.read_csv(chop_tsv + '.tsv', sep='\t', header=0, index_col=None)
        
        else:
            dataread = chop_df
        
        if top is None:
            targets = dataread.iloc[:, 0].values
        else:
            if dataread.shape[0] > top:
                targets = dataread.loc[:, 'Target sequence'].values[:top]
            else:
                targets = dataread.loc[:, 'Target sequence'].values
        # targets = np.array([t[:20] for t in targets])

        results_wt, results_best = [], []
        for target_seq in targets:
            results_df = self.opti(target_seq, opti_type=opti_type, ref=None, offtar_search=offtar_search, mm=mm, dev=dev, 
                                    accepted_mutate=accepted_mutate, cd33cut=cd33cut)
            results_wt.append(results_df.iloc[0, :].values[3:-1])
            results_best.append(results_df.iloc[1, :].values)
            results_df.to_csv('results_{}.csv'.format(target_seq))
        results_wt = np.array(results_wt)
        results_best = np.array(results_best)
        results = np.hstack([results_best[:, :2], results_wt, results_best[:, 2:]])
        col_name = np.append(np.append(results_df.columns[:2], ['WT_' + i for i in results_df.columns[3:-1]]), results_df.columns[2:])
        results_out = pd.DataFrame(results, columns=col_name)
        results_out.to_csv('ResultsSummary.csv')
        return results_out

    def rescore_chopchop(self, chop_tsv=None, chop_df=None, top=None, ref=None, offtar_search=None, mm=6, dev='G0'):
        if chop_tsv is not None:
            if len(chop_tsv) > 4:
                if chop_tsv[-4:] == '.tsv':
                    chop_tsv = chop_tsv[:-4]
            dataread = pd.read_csv(chop_tsv + '.tsv', sep='\t', header=0, index_col=None)
        
        else:
            dataread = chop_df
        
        if top is not None:
            if dataread.shape[0] > top:
                dataread = dataread.iloc[:top, :]
        targets = dataread.loc[:, 'Target sequence'].values

        results = []
        # calculate WT spec score
        for i in range(dataread.shape[0]):
            target = targets[i]
            spec, out_df = self.CasoffinderSpec_(target, target, out_df=True, offtar_search=offtar_search, mm=mm, dev=dev)
            cnt_results = self.score_bin_(out_df.loc[:, 'CRISOT-Score'].values[1:])[-4:]
            if ref is not None:
                ref_merge = ref.loc[:, ['Offtarget_Sequence', 'GUIDE-Seq Reads']]
                ref_merge.columns = [3, 'Reads']
                out_merge = pd.merge(out_df.iloc[1:,:], ref_merge, how='left', on=[3], sort=False)
                out_merge.fillna(0, inplace=True)
                out_merge = out_merge.loc[out_merge.loc[:,'Reads'] > 0, :]
                if out_merge.shape[0] != 0:
                    # out_merge.to_csv('{}_ref_match.tsv'.format(target), sep='\t')
                    spec_ref = self.single_score_(out_merge.loc[:, 'CRISOT-Score'].values[1:], out_cnt=False)
                else:
                    spec_ref = 1.0
            if ref is not None:
                results.append(np.append(np.append(out_df.loc[:, 'CRISOT-Score'].values[0], cnt_results), [spec, spec_ref]))
                results_df = pd.DataFrame(results, columns=['CRISOT-Score', '(0.65, 0.7]', '(0.7, 0.75]', 
                                                    '(0.75, 0.8]', '(0.8, 1.0]', 'CRISOT-Spec', 'ref_Spec'])
            else:
                results.append(np.append(np.append(out_df.loc[:, 'CRISOT-Score'].values[0], cnt_results), [spec]))
                results_df = pd.DataFrame(results, columns=['CRISOT-Score', '(0.65, 0.7]', '(0.7, 0.75]', 
                                                    '(0.75, 0.8]', '(0.8, 1.0]', 'CRISOT-Spec'])

        results_df.iloc[:, :] = np.array(results_df.iloc[:, :].values, dtype='float')

        results_df = pd.concat([dataread, results_df], axis=1)

        rk = np.array(results_df.loc[:, 'CRISOT-Score'].values > self.opti_th, dtype=int)
        results_df['CRISOT_rank'] = rk

        results_out = results_df.sort_values(by=['CRISOT_rank', 'CRISOT-Spec'], ascending=[False, False])
        results_out.index = np.arange(results_out.shape[0])
        results_out.loc[:, 'CRISOT_rank'] = np.arange(1, results_out.shape[0]+1)
        return results_out


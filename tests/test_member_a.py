"""Behavioral boundaries for A features, independent of the frozen dataset."""
import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from common.data import RawData, prediction_requests
from member_a.features import feature_frame, fit_protein_pca, transform_protein_pca


def fixture():
    proteins=[];peptides=[]
    for patient in [1,2]:
        for month in [0,6,12,24]:
            key={'visit_id':f'{patient}_{month}','patient_id':patient,'visit_month':month}
            proteins.extend([{**key,'UniProt':'P1','NPX':float(patient+month)},
                             {**key,'UniProt':'P2','NPX':float(2*patient+month)}])
            peptides.append({**key,'UniProt':'P1','Peptide':'SEQ','PeptideAbundance':float(patient+month)})
    return RawData(pd.DataFrame({'rating':[123.]}),pd.DataFrame(proteins),
                   pd.DataFrame(peptides),pd.DataFrame({'rating':[321.]}))


class MemberAFeatures(unittest.TestCase):
    def test_future_and_clinical_information_do_not_change_past_features(self):
        raw=fixture();visits=raw.proteins[['visit_id','patient_id','visit_month']].drop_duplicates()
        requests=prediction_requests(visits.loc[visits.visit_month<=6])
        args=(['protein::P1','protein::P2'],['peptide::P1::SEQ'])
        expected=feature_frame(requests,raw,*args)
        changed=replace(raw,proteins=raw.proteins.copy(),peptides=raw.peptides.copy(),
                        clinical=pd.DataFrame({'rating':[-999.]}),supplemental=pd.DataFrame())
        changed.proteins.loc[changed.proteins.visit_month>6,'NPX']=1e12
        changed.peptides.loc[changed.peptides.visit_month>6,'PeptideAbundance']=1e12
        pd.testing.assert_frame_equal(expected,feature_frame(requests,changed,*args))
        truncated=replace(changed,proteins=changed.proteins.loc[changed.proteins.visit_month<=6],
                           peptides=changed.peptides.loc[changed.peptides.visit_month<=6])
        pd.testing.assert_frame_equal(expected,feature_frame(requests,truncated,*args))

    def test_validation_only_vocabulary_is_ignored(self):
        raw=fixture();requests=prediction_requests(raw.proteins[['visit_id','patient_id','visit_month']])
        args=(['protein::P1'],['peptide::P1::SEQ'])
        expected=feature_frame(requests,raw,*args)
        extra=raw.proteins.loc[raw.proteins.patient_id==2].assign(UniProt='UNSEEN',NPX=1e12)
        changed=replace(raw,proteins=pd.concat([raw.proteins,extra],ignore_index=True))
        pd.testing.assert_frame_equal(expected,feature_frame(requests,changed,*args))

    def test_empty_current_visit_retains_missing_measurements(self):
        raw=fixture();requests=prediction_requests(pd.DataFrame([{'visit_id':'1_3','patient_id':1,'visit_month':3}]))
        x=feature_frame(requests,raw,['protein::P1'],['peptide::P1::SEQ'])
        self.assertTrue(x.filter(regex='^(protein|peptide)_').isna().all().all())
        self.assertTrue(x.hist_protein_sample_count.eq(1).all())
        self.assertEqual(len(x),16)

    def test_pca_ignores_empty_training_columns_and_repeated_requests(self):
        unique=pd.DataFrame({'visit_id':['a','b','c'], 'protein::P1':[1.,2.,3.],
                             'protein::P2':[4.,7.,2.], 'protein::UNSEEN':[np.nan]*3})
        repeated=pd.concat([unique,unique,unique],ignore_index=True)
        a=fit_protein_pca(unique);b=fit_protein_pca(repeated)
        self.assertNotIn('protein::UNSEEN',a['columns'])
        np.testing.assert_allclose(a['pca'].components_,b['pca'].components_)
        self.assertTrue(np.isfinite(transform_protein_pca(unique,a)).all())


if __name__=='__main__':unittest.main()

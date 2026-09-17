"""Learned vocabulary isolation, offline methods and sparse pipeline behavior."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from text_pandas import as_text
from TextEncodingTransformer import TextEncodingTransformer,embedding_identity,characteristics
from cards import var_text_encode as card
from proxy_data import proxy_data
from roles import Role,RoleMap
pytestmark=pytest.mark.unit


def frame(values):return pd.DataFrame({'review':as_text(pd.Series(values))})


def test_counts_binary_tfidf_unseen_and_sparse_zeros():
    X=frame(['red red blue','blue green','',None]);X.index=[0,0,2,3]
    m=TextEncodingTransformer(['review'],weighting='counts');out=m.fit_transform(X)
    assert out.review__bow__red.tolist()==[2,0,0,0]
    assert out.review__text_missing.tolist()==[0,0,0,1]
    assert all(isinstance(d,pd.SparseDtype) and d.fill_value==0 for d in out.dtypes)
    assert out.index.equals(X.index)
    future=m.transform(frame(['red unseen','unseen','',None]))
    assert all(d.fill_value==0 for d in future.dtypes)
    assert future.review__bow__red.tolist()==[1,0,0,0]
    assert future.review__bow__unknown_fraction.tolist()==[.5,1,0,0]
    assert not any('unseen' in c for c in future)
    b=clone(m).set_params(weighting='binary').fit_transform(X)
    assert b.review__bow__red.iloc[0]==1
    t=clone(m).set_params(weighting='tfidf').fit(X)
    # Missing documents do not inflate the corpus used for IDF; empty strings do.
    vocab=t.models_[('review','bow')]['vectorizer'].vocabulary_
    assert t.models_[('review','bow')]['tfidf'].idf_[vocab['red']]==pytest.approx(np.log(4/2)+1)
    assert not hasattr(clone(m),'models_')
    assert m.transform(X.iloc[:0]).shape==(0,len(out.columns))
    assert m.get_feature_names_out().tolist()==out.columns.tolist()


def test_sparse_proxy_roundtrip_and_downstream_estimator():
    X=frame(['excellent good','awful bad','lovely good','terrible bad'])
    source=proxy_data(X);result=card._analyze(source)
    out=card._apply(source,result,['bow'])
    assert out.clone().equals(out)
    pd.testing.assert_frame_equal(out.pipeline_for_training().fit_transform(out.clean_frame),out.frame)
    pipeline=out.pipeline_for_training();pipeline.steps.append(('model',LogisticRegression()))
    assert pipeline.fit(X,[1,0,1,0]).predict(X).tolist()==[1,0,1,0]
    fitted=pickle.loads(pickle.dumps(pipeline))
    assert fitted.predict(frame(['good unseen']))[0]==1


def test_sentiment_punctuation_negation_and_missing():
    X=frame(['This is good.','This is not good.','This is GOOD!!!','',None])
    out=TextEncodingTransformer(['review'],methods=('sentiment',)).fit_transform(X)
    scores=out.review__sentiment__compound
    assert scores.iloc[0]>0 and scores.iloc[1]<0 and scores.iloc[2]>scores.iloc[0]
    assert scores.iloc[3:].isna().all()
    assert out.review__text_missing.tolist()==[0,0,0,0,1]


def test_characteristics_definitions_and_empty_missing():
    out=TextEncodingTransformer(['review'],methods=('characteristics',)).fit_transform(frame(['Hi 2!!','',None]))
    assert out.review__characteristics__characters.iloc[0]==6
    assert out.review__characteristics__words.iloc[0]==2
    assert out.review__characteristics__exclamations.iloc[0]==2
    assert out.review__characteristics__uppercase_fraction.iloc[0]==pytest.approx(1/6)
    assert out.iloc[1].eq(0).all()
    assert out.filter(like='__characteristics__').iloc[2].isna().all()


def test_lsa_train_vocabulary_rank_and_zero_documents():
    X=frame(['cats dogs pets','cars roads wheels','dogs cats animals','roads cars transport',None,''])
    m=TextEncodingTransformer(['review'],methods=('lsa',),components=100).fit(X)
    assert len(m.models_[('review','lsa')]['names'])==4
    out=m.transform(X)
    assert out.filter(like='component').iloc[4].isna().all()
    assert out.filter(like='component').iloc[5].eq(0).all()
    assert m.transform(frame(['entirelyunknown'])).filter(like='component').eq(0).all().all()
    small=clone(m).fit(frame(['one']))
    assert not small.output_columns_ and not small.removed_columns_
    assert 'requires at least' in small.summary_[0]['Notes']


def test_embedding_mean_repeats_coverage_unknown_and_portability(tmp_path):
    p=tmp_path/'licensed.vec';p.write_text('2 2\ngood 1 0\nbad 0 1\n')
    m=TextEncodingTransformer(['review'],methods=('embedding',),embedding_path=str(p),embedding_digest=embedding_identity(p))
    X=frame(['good good bad unknown','unknown','',None]);out=m.fit_transform(X)
    assert out.review__embedding__dimension_1.iloc[0]==pytest.approx(2/3)
    assert out.review__embedding__coverage.iloc[0]==.75
    assert out.review__embedding__dimension_1.iloc[1:].isna().all()
    assert out.review__embedding__coverage.iloc[1]==0
    assert pd.isna(out.review__embedding__coverage.iloc[3])
    restored=pickle.loads(pickle.dumps(m));p.unlink()
    pd.testing.assert_frame_equal(restored.transform(X),out)


def test_unavailable_resource_malformed_resource_and_combined_retention(tmp_path):
    source=proxy_data(frame(['good product','bad product']))
    result=card._analyze(source)
    out=card._apply(source,result,['sentiment','embedding'])
    assert 'review' in out.columns and 'review__sentiment__compound' in out.columns
    assert 'review' not in card._apply(source,result,['sentiment']).columns
    p=tmp_path/'bad.vec';p.write_text('2 2\ngood NaN 0\nbad 0 1\n')
    m=TextEncodingTransformer(['review'],methods=('embedding',),embedding_path=str(p),embedding_digest=embedding_identity(p)).fit(source.frame)
    assert not m.output_columns_ and 'finite' in m.summary_[0]['Notes']


def test_roles_multiple_columns_and_every_selected_subset_roundtrip():
    X=frame(['great service','bad service','good product','awful product'])
    X['other']=as_text(pd.Series(['red apple','blue sky','green apple','blue sea']))
    X['target']=as_text(pd.Series(['a','b','a','b']))
    X['review__text_missing']=[1,2,3,4]
    roles=RoleMap()
    for c in X:roles.set_roles(c,[Role.TARGET if c=='target' else Role.PREDICTOR])
    source=proxy_data(_df=X,_roles=roles);result=card._analyze(source)
    assert card._predictors(source)==['review','other']
    for selected in [('sentiment',),('characteristics',),('lsa',),('bow','sentiment','lsa','characteristics')]:
        out=card._apply(source,result,selected)
        pd.testing.assert_frame_equal(out.pipeline_for_training().fit_transform(out.clean_frame),out.frame)
        assert out.role_map.roles_for('target')=={Role.TARGET}
        assert out.role_map.roles_for('review__text_missing_2')=={Role.PREDICTOR}
    assert card._apply(source,result,[]) is source


def test_empty_vocabulary_character_analysis_and_stop_words():
    X=frame(['a','a','',None])
    m=TextEncodingTransformer(['review'],stop_words=['a']).fit(X)
    assert not m.output_columns_ and 'empty vocabulary' in m.summary_[0]['Notes']
    m=TextEncodingTransformer(['review'],analyzer='char',ngram_max=2).fit(frame(['Hi!','Oh!']))
    assert '!' in m.models_[('review','bow')]['vectorizer'].vocabulary_
    missing=TextEncodingTransformer(['review']).fit(frame([None,None]))
    assert not missing.output_columns_
    with pytest.raises(ValueError,match='Text semantic'):
        TextEncodingTransformer(['x']).fit(pd.DataFrame({'x':['plain string']}))

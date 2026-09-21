"""DataFrame-preserving text features fitted inside sklearn training pipelines."""
from __future__ import annotations

import copy
import hashlib
import re
import string
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from code_recording import recordable
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted
from text_pandas import is_text

METHODS={'bow':'Bag of words','sentiment':'Sentiment','embedding':'Word embeddings',
    'lsa':'Latent semantics','characteristics':'Text characteristics'}
STATS=('characters','words','mean_word_length','lines','digit_fraction','uppercase_fraction',
       'punctuation_fraction','whitespace_fraction','questions','exclamations')
TOKEN=re.compile(r'(?u)\b\w+\b')
MAX_TEXT_CHARS=10_000_000


@recordable
@lru_cache(maxsize=1)
def sentiment_model():
    from vendor.vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


@recordable
def embedding_identity(path):
    p=Path(path)
    if not p.is_file():raise ValueError('Load a pretrained Word2Vec text file to enable embeddings.')
    if p.stat().st_size>50*1024*1024:raise ValueError('Embedding files are limited to 50 MiB for browser and memory compatibility.')
    return hashlib.sha256(p.read_bytes()).hexdigest()


@recordable
@lru_cache(maxsize=2)
def _load_vectors(path,digest):
    if embedding_identity(path)!=digest:raise ValueError('Embedding resource changed; reload the model.')
    vectors={}
    with open(path,encoding='utf-8') as stream:
        header=stream.readline().split()
        if len(header)!=2 or not all(x.isdigit() for x in header):
            raise ValueError('Expected Word2Vec text format: first line is vocabulary size and dimension. Binary models are unsupported.')
        count,width=map(int,header)
        if not 1<=count<=200000 or not 1<=width<=300:
            raise ValueError('Embedding limits: 200,000 tokens and 300 dimensions.')
        for line in stream:
            pieces=line.split()
            if len(pieces)!=width+1:raise ValueError('Malformed embedding row or inconsistent dimensions.')
            if pieces[0] in vectors:raise ValueError('Duplicate token in embedding resource.')
            vector=np.asarray(pieces[1:],dtype=np.float32)
            if not np.isfinite(vector).all():raise ValueError('Embedding vectors must be finite.')
            vectors[pieces[0]]=vector
            if len(vectors)>count:raise ValueError('Embedding vocabulary exceeds its declared size.')
    if len(vectors)!=count:raise ValueError('Embedding vocabulary count does not match the header.')
    return vectors,width


@recordable
def _texts(series):
    values=[]; missing=[]
    for value in series:
        absent=pd.isna(value)
        if not isinstance(absent,(bool,np.bool_)) or (not absent and not isinstance(value,str)):
            raise ValueError('Text predictors must contain strings or missing values.')
        missing.append(bool(absent));values.append('' if absent else value)
    if sum(map(len,values))>MAX_TEXT_CHARS:raise ValueError('A text column exceeds the 10 million character processing limit.')
    return values,np.asarray(missing,dtype=bool)


@recordable
def characteristics(text):
    words=TOKEN.findall(text);length=len(text);den=max(length,1)
    return [length,len(words),np.mean(list(map(len,words))) if words else 0,
        len(text.splitlines()) if text else 0,sum(c.isdigit() for c in text)/den,
        sum(c.isupper() for c in text)/den,sum(c in string.punctuation for c in text)/den,
        sum(c.isspace() for c in text)/den,text.count('?'),text.count('!')]


@recordable
class TextEncodingTransformer(TransformerMixin,BaseEstimator):
    def __init__(self,columns,*,methods=('bow',),remove_original=True,weighting='tfidf',
                 analyzer='word',ngram_max=1,lowercase=True,strip_accents=False,stop_words=None,
                 min_df=1,max_df=1.0,max_features=500,components=20,lsa_normalize=False,
                 embedding_path='',embedding_digest='',embedding_lowercase=True,embedding_label=''):
        self.columns=columns;self.methods=methods;self.remove_original=remove_original
        self.weighting=weighting;self.analyzer=analyzer;self.ngram_max=ngram_max
        self.lowercase=lowercase;self.strip_accents=strip_accents;self.stop_words=stop_words
        self.min_df=min_df;self.max_df=max_df;self.max_features=max_features
        self.components=components;self.lsa_normalize=lsa_normalize
        self.embedding_path=embedding_path;self.embedding_digest=embedding_digest
        self.embedding_lowercase=embedding_lowercase;self.embedding_label=embedding_label

    def _validate(self,X):
        if not isinstance(X,pd.DataFrame) or not X.columns.is_unique:
            raise ValueError('Text encoding requires a DataFrame with unique column names.')
        if any(c not in X for c in self.columns):raise ValueError('A source text column is missing.')

    def _vectorizer(self):
        return CountVectorizer(analyzer=self.analyzer,ngram_range=(1,int(self.ngram_max)),
            lowercase=self.lowercase,strip_accents='unicode' if self.strip_accents else None,
            stop_words=self.stop_words if self.analyzer=='word' else None,
            min_df=int(self.min_df),max_df=float(self.max_df),max_features=int(self.max_features),
            binary=self.weighting=='binary',token_pattern=r'(?u)\b\w+\b' if self.analyzer=='word' else None)

    def fit(self,X,y=None):
        self._validate(X)
        if any(m not in METHODS for m in self.methods):raise ValueError('Unknown text encoding method.')
        if self.weighting not in ('tfidf','counts','binary'):raise ValueError('Unknown term weighting.')
        if self.analyzer not in ('word','char') or not 1<=self.ngram_max<=3:raise ValueError('Unsupported tokenizer settings.')
        if not 1<=self.max_features<=4096 or not 1<=self.components<=300:raise ValueError('Vocabulary/component limits exceeded.')
        self.feature_names_in_=np.asarray(X.columns,dtype=object);self.n_features_in_=len(X.columns)
        self.models_={};self.summary_=[];self.details_={m:[] for m in self.methods};self.missing_names_={}
        used=set(X.columns)
        def name(base):
            value=base;n=2
            while value in used:value=f'{base}_{n}';n+=1
            used.add(value);return value
        for column in dict.fromkeys(self.columns):
            if not is_text(X[column]):raise ValueError(f'{column!r} requires the Text semantic type during fitting.')
            texts,missing=_texts(X[column]);valid=[t for t,absent in zip(texts,missing) if not absent]
            self.missing_names_[column]=name(f'{column}__text_missing')
            for method in dict.fromkeys(self.methods):
                row={'Source':str(column),'Method':METHODS[method],'Documents':len(texts),
                     'Missing':int(missing.sum()),'Empty':sum(not t.strip() for t in valid),'Outputs':0,
                     'Generated columns':'','Notes':''}
                try:
                    if not valid or not any(t.strip() for t in valid) and method!='characteristics':
                        raise ValueError('No nonempty documents; no features generated.')
                    model={}
                    if method in ('bow','lsa'):
                        vectorizer=self._vectorizer();matrix=vectorizer.fit_transform(valid)
                        terms=vectorizer.get_feature_names_out()
                        model={'vectorizer':vectorizer,'tfidf':None}
                        if method=='lsa' or self.weighting=='tfidf':
                            model['tfidf']=TfidfTransformer();matrix=model['tfidf'].fit_transform(matrix)
                        if method=='bow':
                            suffixes=[str(t) for t in terms]+['unknown_fraction']
                            counts=vectorizer.transform(valid)
                            frequencies=np.asarray((counts>0).sum(axis=0)).ravel()
                            order=np.argsort(-frequencies,kind='stable')[:100]
                            self.details_[method].extend({'Source':str(column),'Term':str(terms[i]),'Document frequency':int(frequencies[i])} for i in order)
                        else:
                            width=min(int(self.components),matrix.shape[0]-1,matrix.shape[1]-1)
                            if width<1:raise ValueError('LSA requires at least two nonmissing documents and two vocabulary terms.')
                            model['svd']=TruncatedSVD(n_components=width,random_state=2025)
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore',RuntimeWarning)
                                model['svd'].fit(matrix)
                            suffixes=[f'component_{i+1}' for i in range(width)]
                            for i,weights in enumerate(model['svd'].components_):
                                positive=[str(terms[j]) for j in np.argsort(-weights) if weights[j]>0][:5]
                                negative=[str(terms[j]) for j in np.argsort(weights) if weights[j]<0][:5]
                                variance=model['svd'].explained_variance_ratio_[i]
                                self.details_[method].append({'Source':str(column),'Component':i+1,
                                    'Explained variance':round(float(variance),4) if np.isfinite(variance) else None,
                                    'Positive terms':', '.join(positive),'Negative terms':', '.join(negative)})
                            row['Notes']='Components describe associations, not named topics; dimension capped by training data.'
                    elif method=='sentiment':
                        sentiment_model();suffixes=['compound','positive','negative','neutral']
                        row['Notes']='VADER 3.3.2, English; fixed lexicon and rules. Not reliable for sarcasm or unsupported languages.'
                    elif method=='embedding':
                        if not self.embedding_path or not self.embedding_digest:raise ValueError('Load a pretrained Word2Vec text file to enable embeddings.')
                        vectors,width=_load_vectors(self.embedding_path,self.embedding_digest)
                        # Copy only vocabulary needed for training? No: unseen training tokens may exist in pretrained resource.
                        model={'vectors':vectors,'width':width}
                        suffixes=[f'dimension_{i+1}' for i in range(width)]+['coverage']
                        row['Notes']=f'Mean word vectors; {self.embedding_label or Path(self.embedding_path).name}; SHA256 {self.embedding_digest[:12]}. Verify model language and license.'
                    else:
                        suffixes=list(STATS)
                        row['Notes']='Unicode word tokens; punctuation is ASCII; uppercase and other fractions use all characters as denominator.'
                    model['names']=[name(f'{column}__{method}__{s}') for s in suffixes]
                    self.models_[(column,method)]=model
                    row['Outputs']=len(suffixes);row['Generated columns']=', '.join(model['names'])
                except (ValueError,OSError,UnicodeError) as error:
                    row['Notes']=str(error)
                self.summary_.append(row)
        self._outputs()
        return self

    def _outputs(self):
        active={c for c,m in self.models_}
        self.output_columns_=[n for model in self.models_.values() for n in model['names']]+[self.missing_names_[c] for c in self.columns if c in active]
        # A failed selected method retains the source for inspection or another card.
        self.removed_columns_=[c for c in self.columns if self.remove_original and self.methods and all((c,m) in self.models_ for m in self.methods)]
        if len(self.output_columns_)>8192:raise ValueError('Text encoding exceeds 8192 total features. Reduce vocabulary size or selected methods.')

    def select(self,methods,remove_original=True):
        """Use the preview's fitted results without refitting; sklearn cloning drops them."""
        result=copy.copy(self);result.methods=tuple(m for m in METHODS if m in methods);result.remove_original=remove_original
        result.models_={key:value for key,value in self.models_.items() if key[1] in methods}
        result.summary_=[r for r in self.summary_ if r['Method'] in {METHODS[m] for m in methods}]
        result._outputs();return result

    def transform(self,X):
        check_is_fitted(self,'models_');self._validate(X)
        if set(self.output_columns_)&set(X.columns):raise ValueError('A generated text feature name already exists.')
        parts=[X.drop(columns=self.removed_columns_).copy()]
        cache={c:_texts(X[c]) for c in self.columns}
        # Mixed sparse/dense DataFrames may be densified by downstream sklearn estimators.
        dense_bytes=len(X)*(len(self.output_columns_)+len(X.columns))*8
        if dense_bytes>128*1024*1024:raise ValueError('Potential dense text output exceeds 128 MiB; reduce vocabulary, components or selected methods.')
        for (column,method),model in self.models_.items():
            texts,missing=cache[column]
            if method in ('bow','lsa'):
                counts=model['vectorizer'].transform(texts)
                matrix=model['tfidf'].transform(counts) if model['tfidf'] is not None and len(X) else counts
                if method=='bow':
                    analyzer=model['vectorizer'].build_analyzer();vocab=model['vectorizer'].vocabulary_
                    ratios=[]
                    for t in texts:
                        tokens=analyzer(t);ratios.append(sum(token not in vocab for token in tokens)/len(tokens) if tokens else 0.)
                    matrix=sparse.hstack([matrix,sparse.csr_matrix(np.asarray(ratios).reshape(-1,1))],format='csr')
                    if matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes>128*1024*1024:
                        raise ValueError('Sparse term output exceeds 128 MiB; reduce the vocabulary.')
                    # pandas versions differ in the implicit float fill value; term absence is always zero.
                    block=pd.DataFrame.sparse.from_spmatrix(matrix,index=X.index,columns=model['names'])
                    block=pd.DataFrame({name:pd.arrays.SparseArray(series.array.sp_values,
                        sparse_index=series.array.sp_index,fill_value=0.0,dtype=pd.SparseDtype(float,0.0))
                        for name,series in block.items()},index=X.index)
                    parts.append(block)
                    continue
                values=model['svd'].transform(matrix) if len(X) else np.empty((0,len(model['names'])))
                if self.lsa_normalize and len(X):values=normalize(values)
            elif method=='sentiment':
                values=np.full((len(X),4),np.nan)
                for i,t in enumerate(texts):
                    if t.strip():
                        scores=sentiment_model().polarity_scores(t)
                        values[i]=[scores['compound'],scores['pos'],scores['neg'],scores['neu']]
            elif method=='embedding':
                values=np.full((len(X),model['width']+1),np.nan)
                for i,t in enumerate(texts):
                    tokens=TOKEN.findall(t.lower() if self.embedding_lowercase else t)
                    known=[model['vectors'][token] for token in tokens if token in model['vectors']]
                    values[i,-1]=len(known)/len(tokens) if tokens else 0
                    if known:values[i,:-1]=np.mean(known,axis=0)
            else:
                values=np.asarray([characteristics(t) for t in texts],dtype=float).reshape(len(X),len(STATS))
            values[missing,:]=np.nan
            parts.append(pd.DataFrame(values,index=X.index,columns=model['names']))
        active={c for c,m in self.models_}
        for c in self.columns:
            if c in active:
                values=cache[c][1].astype(np.int8)
                if (c,'bow') in self.models_:values=pd.arrays.SparseArray(values.astype(float),fill_value=0.0)
                parts.append(pd.DataFrame({self.missing_names_[c]:values},index=X.index))
        return pd.concat(parts,axis=1)

    def get_feature_names_out(self,input_features=None):
        check_is_fitted(self,'models_')
        return np.asarray([c for c in self.feature_names_in_ if c not in self.removed_columns_]+self.output_columns_,dtype=object)

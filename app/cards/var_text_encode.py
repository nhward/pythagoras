"""Text feature panels and training-safe numeric encoding."""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

if __name__=='__main__':
    ROOT=Path(__file__).resolve().parent.parent
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
from card import Card
from module import Module
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny import render, req, ui
from text_pandas import as_text, is_text
from TextEncodingTransformer import METHODS, TextEncodingTransformer, embedding_identity

GUIDES={
 'bow':'Vocabulary learned from training documents. Counts preserve repeats; binary records presence; TF–IDF reweights by document frequency and normalizes rows. The detail table shows up to 100 leading terms per source. Unknown test terms are ignored and their fraction is exported. Empty and missing text give zero term vectors; a separate missing-text indicator distinguishes them. Outputs use sparse numeric columns.',
 'sentiment':'Fixed English VADER 3.3.2 lexicon and rules, bundled for offline use. Exports compound, positive, negative and neutral scores. Original capitalization and punctuation are preserved. Empty and missing text have missing scores. The table shows score distributions and examples. Sarcasm, domain meanings and other languages can be misleading.',
 'embedding':'Upload a pretrained Word2Vec text file in Settings. First line: token count and dimension, followed by token and numeric-vector rows. Exports the mean recognized word vector and token coverage. Repeated words contribute repeatedly. Unknown words are ignored; documents with no known words have missing vector coordinates. No model is downloaded automatically. Verify the model license and language. Uploads must be supplied again after bookmark restoration or moving the unfitted pipeline to another environment.',
 'lsa':'TF–IDF followed by TruncatedSVD, fitted inside training folds. Components are capped by available documents and vocabulary. Displays explained variance and strongest positive/negative terms. Components are associations, not named topics; signs and axes can change between fits. Sparse term matrices remain sparse until the compact component output.',
 'characteristics':'Non-learned counts of characters, Unicode word tokens, lines, questions and exclamations; mean word length and proportions of digits, uppercase letters, ASCII punctuation and whitespace. Fractions use all characters as denominator. Empty text yields zeros; missing text remains missing. Distributions and example measurements are shown.'}

@dataclass
class AnalysisResult:
    model:TextEncodingTransformer|None=None
    frame:pd.DataFrame|None=None
    error:str=''


def _predictors(source):
    return [c for c in source.columns if Role.PREDICTOR in source.role_map.roles_for(c)
        and Role.TARGET not in source.role_map.roles_for(c) and not str(c).startswith(Card.SHADOW_PREFIX)
        and is_text(source.frame[c])]


def _analyze(source,**options):
    columns=_predictors(source)
    if not columns:return AnalysisResult()
    try:
        model=TextEncodingTransformer(columns,methods=tuple(METHODS),remove_original=False,**options)
        return AnalysisResult(model,model.fit_transform(source.frame))
    except (ValueError,TypeError,OSError) as error:return AnalysisResult(error=str(error))


def _apply(source,result,selected,remove_original=True):
    if result.error or result.model is None or not selected:return source
    model=result.model.select(selected,remove_original)
    if not model.output_columns_:return source
    columns=[c for c in source.columns if c not in model.removed_columns_]+model.output_columns_
    frame=result.frame.loc[:,columns].copy()
    for c,name in model.missing_names_.items():
        if name in frame and (c,'bow') not in model.models_ and isinstance(frame[name].dtype,pd.SparseDtype):
            frame[name]=frame[name].sparse.to_dense().astype(np.int8)
    roles=RoleMap()
    for c in model.output_columns_:roles.set_roles(c,[Role.PREDICTOR])
    return source.with_pipeline_step(model,name='var_text_encode',operation='Encode text predictors',
        preview_frame=frame,added_roles=roles,removed_columns=model.removed_columns_)


def _detail(result,method):
    if result.model is None:return pd.DataFrame()
    if method in ('bow','lsa'):return pd.DataFrame(result.model.details_[method])
    rows=[]
    for (source,kind),model in result.model.models_.items():
        if kind!=method:continue
        names=model['names']
        if method=='embedding':names=[names[-1]]
        for name in names:
            values=result.frame[name].dropna()
            rows.append({'Source':str(source),'Feature':name,'Available':len(values),
                'Minimum':round(float(values.min()),4) if len(values) else None,
                'Median':round(float(values.median()),4) if len(values) else None,
                'Maximum':round(float(values.max()),4) if len(values) else None,
                'First values':', '.join(f'{v:.4g}' for v in values.iloc[:5])})
    return pd.DataFrame(rows)


def instance():
    this=Card(file=__file__,mutable=True)
    this.exclude_configuration_input('EmbeddingFile')
    this.long_name='Text encoding'
    this.description='Generate numeric text predictors using vocabulary, sentiment, embeddings, latent semantics and text characteristics.'
    
    this.front=lambda: ui.navset_bar(*[
        ui.nav_panel(label,ui.output_ui(f'Message_{method}'),
            ui.output_data_frame(f'Summary_{method}',guide=this,title=label+' preview',position='left',text=GUIDES[method]),
            ui.output_data_frame(f'Detail_{method}')) for method,label in METHODS.items()],
        id='EncodingType',selected='Bag of words',title=None,padding=0,fillable=True)
    
    this.back=lambda:ui.TagList(
        ui.card_header(id='Text encoding audit',class_='text-primary text-center'),
        ui.output_ui(id='AuditSummary'),
        ui.output_data_frame(
            id='AuditTable',
            guide=this,title='Outgoing text features',position='left',
            text='Enabled methods only. Lists generated names and counts, document availability, skipped methods and source retention. Each source also gets one missing-text indicator when any method succeeds. A source is removed only if every selected method succeeds for it. Counts describe the preview; training folds learn their own vocabularies and LSA dimensions.'
        )
    )

    this.footer=lambda: ui.TagList(
        ui.input_checkbox_group(id='Encode',label='Encode',choices=METHODS,selected=[],inline=True,
        guide=this,title='Apply text features',position='top',text='Choose any combination independently of the active tab. All methods read the original source text. Outputs are numeric Predictors and require no Basket encoding. Adds one sklearn step; uncheck all to restore incoming data. Unavailable methods add nothing; successful methods still apply. No Target or observation importance is used.'),
        ui.output_ui('Busy'),ui.output_text('Status')
    )
    
    this.settings=lambda: ui.TagList(
        ui.input_checkbox(id='RemoveOriginal',label='Remove original variables',value=True,guide=this,position='left',
            text='Remove a text source only if all selected methods successfully produce outputs for it. Otherwise retain it for inspection. Retained text may need removal before fitting a numeric-only estimator. A missing-text indicator accompanies generated features.'),
        ui.hr(),
        ui.h6('Vocabulary and latent semantics'),
        ui.input_select(id='Weighting',label='Bag-of-words weighting',choices={'tfidf':'TF–IDF','counts':'Counts','binary':'Binary presence'},selected='tfidf',guide=this,position='left',text='TF–IDF learns inverse document frequencies from nonmissing training documents and normalizes each row. Counts preserve repeated tokens; binary records presence. LSA always uses TF–IDF; the binary option also binarizes its initial counts.'),
        ui.input_select(id='Analyzer',label='Token units',choices={'word':'Words','char':'Characters'},selected='word',guide=this,position='left',text='Words are Unicode word-character sequences, including single-character words. Characters allow subword patterns without language-specific tokenization. This affects vocabulary and LSA only.'),
        ui.input_numeric(id='NgramMax',label='Maximum n-gram length',value=1,min=1,max=3,step=1,guide=this,position='left',text='Include single units through this many consecutive units. Word bigrams capture short phrases. Character n-grams capture spelling fragments. Longer ranges increase vocabulary and memory.'),
        ui.input_checkbox(id='Lowercase',label='Lowercase vocabulary text',value=True,guide=this,position='left',text='Lowercase only the vocabulary/LSA input. Sentiment and text characteristics always use the original text.'),
        ui.input_checkbox(id='StripAccents',label='Normalize accents',value=False,guide=this,position='left',text='Unicode accent normalization for vocabulary/LSA only; may merge words with distinct meanings.'),
        ui.input_text_area(id='StopWords',label='Stop words (one per line)',value='',guide=this,position='left',text='Optional explicit language-specific words to omit, one per line. Empty means no stop-word filtering. Apply the same casing/normalization as the tokenizer. Ignored for character analysis.'),
        ui.input_numeric(id='MinDF',label='Minimum document count',value=1,min=1,step=1,guide=this,position='left',text='Keep terms appearing in at least this many nonmissing training documents. Counts are unweighted; an empty vocabulary is reported and leaves the source unchanged.'),
        ui.input_slider(id='MaxDF',label='Maximum document fraction',min=0.1,max=1.0,value=1.0,step=0.05,guide=this,position='left',text='Exclude terms found in more than this fraction of nonmissing training documents. One disables this filter.'),
        ui.input_numeric(id='MaxFeatures',label='Maximum vocabulary size',value=500,min=1,max=4096,step=1,guide=this,position='left',text='Vocabulary limit per source and vocabulary method. Term outputs stay sparse. Total output is limited to 8192 features; dense and sparse output blocks are limited to 128 MiB. Each source is limited to 10 million characters. No silent row sampling or truncation.'),
        ui.input_numeric(id='Components',label='LSA components',value=20,min=1,max=300,step=1,guide=this,position='left',text='Requested compact dimensions, capped below both training document count and vocabulary size. Requires at least two documents and two terms. Each validation fold refits its decomposition with seed 2025.'),
        ui.input_checkbox(id='LSANormalize',label='Normalize LSA vectors',value=False,guide=this,position='left',text='Scale each nonzero document component vector to length one after projection. Missing documents remain missing; empty documents give zeros.'),
        ui.hr(),
        ui.h6('Pretrained word embeddings'),
        ui.input_file(id='EmbeddingFile',label='Word2Vec text model',accept=['.txt','.vec'],multiple=False,guide=this,position='left',text='Upload a UTF-8 Word2Vec text file with count/dimension header. No binary models or automatic downloads. Limits: 50 MiB, 200,000 tokens, 300 dimensions. Use a model you have permission to use. Files must be supplied again after bookmark restoration; fitted pipelines contain vectors, while unfitted recipes require the original resource.'),
        ui.input_checkbox(id='EmbeddingLowercase',label='Lowercase embedding tokens',value=True,guide=this,position='left',text='Match the casing used by the pretrained vocabulary. Mean pooling uses every recognized token occurrence. Exported coverage is recognized tokens divided by all tokens; unknown tokens are ignored. No recognized tokens means missing vector coordinates.')
    )

    def server(input,output,session):

        busy=this.busy()

        @this.settle(2)
        @this.suspendable(calc=True)
        def Options():
            upload=input.EmbeddingFile() or []
            path=upload[0]['datapath'] if upload else ''
            # Digest check is small and local; no resource download is performed.
            try:digest=embedding_identity(path) if path else ''
            except (ValueError,OSError):digest='invalid'
            return {
                'weighting': input.Weighting(),
                'analyzer': input.Analyzer(),
                'ngram_max': int(input.NgramMax() or 1),
                'lowercase': bool(input.Lowercase()),
                'strip_accents': bool(input.StripAccents()),
                'stop_words': [w.strip() for w in (input.StopWords() or '').splitlines() if w.strip()] or None,
                'min_df': int(input.MinDF() or 1),
                'max_df': float(input.MaxDF()),
                'max_features': int(input.MaxFeatures() or 500),
                'components': int(input.Components() or 20),
                'lsa_normalize': bool(input.LSANormalize()),
                'embedding_path': path,
                'embedding_digest': digest,
                'embedding_label': upload[0]['name'] if upload else '',
                'embedding_lowercase': bool(input.EmbeddingLowercase())
            }


        @busy.track('Preparing text features…')
        @this.extended_task
        async def Calculate(source, options):
            result=_analyze(source, **options) if Module.IS_SHINYLIVE else await asyncio.to_thread(_analyze,source, **options)
            return source,options,result


        @this.suspendable()
        def Start():
            Calculate.cancel()
            Calculate.invoke(this.input_data().clone(), Options())


        @this.suspendable(calc=True)
        def Analysis():
            source,options,result=Calculate.result()
            req(source.equals(this.input_data()) and options==Options())
            return result


        @this.suspendable(calc=True)
        def Export():
            source=this.input_data()
            selected=tuple(input.Encode() or [])
            if not selected:
                return source
            return _apply(source,Analysis(), selected, bool(input.RemoveOriginal()))


        def register(method):

            @output(id=f'Message_{method}')
            @render.ui
            def message():
                result=Analysis()
                if result.error:
                    return ui.p(result.error, class_='text-danger')
                if result.model is None:
                    return ui.p('No Text predictors are available.')
                rows=[r for r in result.model.summary_ if r['Method']==METHODS[method]]
                if not any(r['Outputs'] for r in rows):return ui.p(' '.join(dict.fromkeys(r['Notes'] for r in rows)), class_='text-warning')
                return ui.p(GUIDES[method])


            @output(id=f'Summary_{method}')
            @render.data_frame
            def summary():
                result=Analysis()
                rows=[r for r in result.model.summary_ if r['Method']==METHODS[method]] if result.model else []
                return render.DataTable(pd.DataFrame(rows), width='100%', height='auto')


            @output(id=f'Detail_{method}')
            @render.data_frame
            def detail():
                return render.DataTable(_detail(Analysis(),method), width='100%', height='auto')


        for method in METHODS:register(method)


        @output
        @render.text
        def Status():
            selected=input.Encode() or []
            if not selected:
                return None
            result=Analysis()
            if result.error:
                return result.error
            if result.model is None:
                return None
            model=result.model.select(selected, bool(input.RemoveOriginal()))
            failed=sum(r['Outputs']==0 for r in model.summary_)
            if failed:
                return f'{failed} unavailable source/method combinations; see the audit.'
            return f'{len(model.output_columns_)} new predictors'


        @output
        @render.ui
        def AuditSummary():
            source=this.input_data()
            out=Export()
            count=lambda data:sum(Role.PREDICTOR in data.role_map.roles_for(c) for c in data.columns)
            return ui.p(f'Predictors: {count(source)} → {count(out)}, Originals removed: {len(set(source.columns)-set(out.columns))}')


        @output
        @render.data_frame
        def AuditTable():
            selected=input.Encode() or []
            rows=[]
            if selected:
                result=Analysis()
                if result.model:
                    model=result.model.select(selected, bool(input.RemoveOriginal()))
                    for row in model.summary_:
                        item=dict(row)
                        item['Original retained?']='No' if any(str(c)==row['Source'] for c in model.removed_columns_) else 'Yes'
                        rows.append(item)
                    active={c for c,m in model.models_}
                    for c in model.columns:
                        if c in active:
                            rows.append({
                                'Source':str(c),
                                'Method':'Missing text indicator',
                                'Outputs':1,
                                'Generated columns':model.missing_names_[c]
                            })
            return render.DataTable(pd.DataFrame(rows), width='100%', height='auto')


        @output
        @render.ui
        def Busy():
            return busy.ui()
        
        session.on_ended(Calculate.cancel)
        
        return Export
    this.server=server
    
    return this


if Module.running_directly(name=__name__):
    this=instance()
    frame=pd.DataFrame({'review':as_text(pd.Series(['I love this excellent product!','This is terrible and disappointing.',
        'The parcel arrived on Tuesday.','Excellent service and a lovely product.', '',None]))})
    this._imports.set(proxy_data(_df=frame,_name='Text encoding example'))
    this.run()

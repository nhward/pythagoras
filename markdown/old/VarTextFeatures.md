#### Goal
The goal with this card is to appropriately convert text variables into numeric ones so that information from text variables can participate in the analysis and modeling.
A text variable is defined as a character variable that has high cardinality i.e. there are few, if any repetitions. It is normal for text variables to be multi-word, multi-sentence and even multi-paragraph. 

#### Simple features
The simple features produces the following numeric evaluations. Each of these becomes a new feature (variable) in the dataset. It is possible that some features might be zero throughout.

Since punctuation, word separation and letter-case are not universal, these simple features will be of little use for _non-segmenting_ languages like Chinese. 

 - **n_words** - Number of words.
 - **n_uq_words** - Number of unique words.
 - **n_charS** - Number of characters. Not counting urls, hashtags, mentions or white spaces.
 - **n_uq_charS** - Number of unique characters. Not counting urls, hashtags, mentions or white spaces.
 - **n_digits** - Number of digits.
 - **n_hashtags** - Number of hashtags, word preceded by a '#'.
 - **n_uq_hashtags** - Number of unique hashtags, word preceded by a '#'.
 - *n_mentions** - Number of mentions, word preceded by a '@'.
 - **n_uq_mentions** - Number of unique mentions, word preceded by a '@'.
 - **n_commas** - Number of commas.
 - **n_periods** - Number of periods.
 - **n_exclaims** - Number of exclamation points.
 - **n_extraspaces** - Number of times more then 1 consecutive space have been used.
 - **n_caps** - Number of upper case characters.
 - **n_lowers** - Number of lower case characters.
 - **n_urls** - Number of urls.
 - **n_uq_urls** - Number of unique urls.
 - **n_nonasciis** - Number of non ascii characters.
 - **n_puncts** - Number of punctuations characters, not including exclamation points, periods and commas.
 - **first_person** - Number of "first person" words: I, me, myself, my, mine, this.
 - **first_personp** - Number of "first person plural" words: we, us, our, ours, these
 - **second_person** - Number of "second person" words: you, yours, your, yourself
 - **second_personp** - Number of "second person plural" words: he, she, it, its, his, hers
 - **third_person** - Number of "third person" words: they, them, theirs, their, they're, their's, those, that
 - **to_be** - Number of "to be" words: am, is, are, was, were, being, been, be, were, be
 - **prepositions** - Number of preposition words: about, below, excepting, off, toward, above, beneath, on, under, across, from, onto, underneath, after, between, in, out, until, against, beyond, outside, up, along, but, inside, over, upon, among, by, past, around, concerning, regarding, with, at, despite, into, since, within, down, like, through, without, before, during, near, throughout, behind, except, of, to, for

The simple-encoding chart shows the numeric features as boxplots with a common numeric axis.

#### Sentiment
Sentiment is the process of assigning numeric degrees of sentiment (e.g. positive value for a positive sentiment versus negative value for a negative sentiment) to words. The sentiments are summed over the text. A sum that is zero suggests either that no sentimental words were used or that the positive and negative contributions have cancelled out.
Many words will be neutral and have no sentiment value associated with them.
The sentiment lexicon in use is "afinn". It is only for English text. The lexicon consists of 2,477 rows of 2 variables:

 * **word** - An English word
 * **score** - Indicator for sentiment: integer between -5 and +5
 
Bear in mind that for long text strings sentiment may not be uniform across its length.

The chart for sentiment encoding is a histogram of sentiment.

#### Hash encoding
Hash encoding adds a number of hash-encoded variables. The new features have little interpretability except that they may capture some essence of the text in a numeric form. 
The goal is to keep the number of new hash variables to a manageable number while ensuring the "collision rate" is not too high. A collision is when two different observations match for each of its hash variables. In other words, the text(s) differ but the numbers derived from them them do not.

The only way to tell whether hash encoding is useful is to check whether the hash variables are important predictors. Hash encoding is suited to very high cardinalities with no expectation of repetition. It is normal to combine any text variables into a single variable before hash encoding as this keeps the number of new features as low as possible. The collision rate is displayed in the card's footer.

The chart for hash encoding is a heat map of the hash values over the span of observations and text-variables. 

#### Target encoding
Target encoding requires the target role to be assigned to a variable. The target must be a numeric variable or binary categorical variable.
Target encoding requires __some__ repetition of text values. Target encoding is not a single algorithm. The choices are:

 * **bayes** - `embed::step_lencode_bayes()` creates a specification of a recipe step that will convert a nominal (i.e. factor) predictor into a single set of scores derived from a generalized linear model estimated using Bayesian analysis. 
 * **glm** - `embed::step_lencode_glm()` creates a specification of a recipe step that will convert a nominal (i.e. factor) predictor into a single set of scores derived from a generalized linear model.
 * **mixed** - `embed::step_lencode_mixed()` creates a specification of a recipe step that will convert a nominal (i.e. factor) predictor into a single set of scores derived from a generalized linear mixed model.
 * **binary** - `embed::step_woe()` creates a specification of a recipe step that will transform nominal data into its numerical transformation based on _weights of evidence_ against a binary outcome.

The chart for target encoding is a histogram of the encoding.
 
#### Actions

Below are some actions that follow from a better understanding of the text encoding process.

 - Whether to combine multiple text variables into a single combined text value. 
 - Whether to add simple numeric text features to the dataset.
 - Whether to add sentiment features to the dataset.
 - Whether to add hashed features to the dataset.
 - Whether to replace text variables with their numeric target encoding.
 - Whether to remove any constant or near-constant variables (i.e. Near-Zero Variance) that may have been produced.

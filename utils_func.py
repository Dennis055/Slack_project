import re
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
# nltk.download('wordnet')


def keyword_collect(keyword):
    word = []
    score = []
    for i in model.most_similar(keyword):
        word.append(i[0])
        score.append(model[i[0]])
    return word , score

def plot_word2vec(X_reduced , total_word , all_word):
    colors = sns.color_palette()
    plt.figure(figsize = (12,6))
    plt.title('word2vec 視覺化結果', fontsize= 18)
    temp_df = pd.DataFrame(X_reduced)
    temp_df['word'] = pd.Series(total_word)
    temp_df['type'] = pd.Series(all_word)
    ax = sns.scatterplot(temp_df[0] ,temp_df[1] ,hue = 'type',data=temp_df ,color = colors[1])
    for i in range(len(temp_df)):
        ax.text(temp_df[0][i] , temp_df[1][i] , temp_df['word'][i] , fontsize = 12)

def build_lda_model(tidy_text_segment):
    id2word = gensim.corpora.Dictionary(tidy_text_segment);
    corpus = [id2word.doc2bow(text) for text in tidy_text_segment];
    lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics);
    return lda


def get_lda_topics(model, num_topics = 10):
    word_dict = {};
    for i in range(num_topics):
        words = model.show_topic(i, topn = 20);
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words];
    return pd.DataFrame(word_dict);

def get_this_topic(resource , num_topics):
    lda = build_lda_model(resource)
    return get_lda_topics(lda , num_topics)


# Helper function
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=0) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
    
def exe_freq_word(segment):
    # Initialise the count vectorizer with the English stop words
    count_vectorizer = CountVectorizer()
    # Fit and transform the processed titles
    count_data = count_vectorizer.fit_transform(segment)
    print('Top 10 most common words')
    # Visualise the 10 most common words
    plot_10_most_common_words(count_data, count_vectorizer)


def lemmatize_stemming(text):
    stemmer = SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

def transform_data_from_md_to_json(path , slack_channel):
    print('processing {} channel'.format(slack_channel))

    file_path = path + slack_channel + '.md'

    fh = open(file_path)
    for line in fh:
        data=fh.read()
    fh.close()

    data_l = data.split('\n')
    # data_l = [x for x in data_l if  '**'  in x]
    data_l = [x for x in data_l if  'AM' in x or 'PM' in x]
    data_l = [x.rstrip() for x in data_l  if len(x) > 2]
    print(len(data_l))
    # filter chatbot 
    data_l = [x for x in data_l if  'WordPress commit' not in x ]
    data_l = [x for x in data_l if  'WordPress Trac' not in x ]
    print(len(data_l))

    data_json = []

    start_p =[] 
    end_p = []
    all_data = []
    for idx ,i  in enumerate( data_l):
        data = {}
        data['time'] = i[0:9]
        data['text'] = i[9:]

        if '&lt;meeting' in i:
            start_p.append(idx)
        elif '&lt;/meeting' in i :
            end_p.append(idx)

        json_data = json.dumps(data)
        data_json.append(json_data)
        all_data.append(data)
    return all_data


def filter_user(text):
#     s = '**maedahbatool**: <@U28P43GKB> Do we have a n..'
    try:
        account = re.findall('\*+[\S]+\*+',text)[0].replace('*','')
    except:
        account = ''
    return account

def remove_user(text):
#     s = '**maedahbatool**: <@U28P43GKB> Do we have a n..'
    try:
        account = re.findall('\*+[\S]+\*+',text)[0]
    except:
        account = ''
    clean_text = text.replace(account , '').replace(':','')
    
    return clean_text
    
#讀入我們會用到的套件
import re
import jieba.posseg as pseg
import pandas as pd
import datetime
import jieba
from datetime import datetime
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True) #為了能在本地端調用
import plotly.tools as tls
import plotly.figure_factory as ff
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import swifter
import numpy as np
import itertools
from scipy.sparse import csr_matrix
# libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True) #為了能在本地端調用
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import cufflinks
cufflinks.go_offline(connected=True)
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
# Code Snippet for Creating LDA visualization
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import nltk
from nltk.corpus import stopwords
import gensim
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
import pyLDAvis
# Visualize the topics
from pyLDAvis import sklearn as sklearn_lda
import pyLDAvis.gensim
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import en_core_web_sm
from spacy import displacy
from textstat import flesch_reading_ease
from hanziconv import HanziConv
import jieba
import re
import numpy as np
import itertools
from scipy.sparse import csr_matrix


jieba.load_userdict('/Users/Dennis/Python與商業分析/R_text_mining/my_dict.txt')


def get_custom_stopwords(stop_words_file):
    with open(stop_words_file,'r' , encoding='gbk' , newline= '') as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i.replace('\r','') for i in stopwords_list]
    return custom_stopwords_list


stop_words_file = '/Users/Dennis/data_science/NLP /Term project/哈工大停用词表.txt'


stopwords = get_custom_stopwords(stop_words_file)
new_stop_words = []
for word in stopwords:
    new_stop_words.append(HanziConv.toTraditional(word))

    
stopwords =  set(new_stop_words)
# 前處理函式
def generate_line_sentence(all_text):  
    # 以句號分句
    all_text = re.split('[。，！？!?,]', all_text)
    
    # 去除所有非中文字元
#     all_text = [_get_chinese(s) for s in all_text]
    
    return all_text

def train_lda_objects(text , topic_num , stop):
#     nltk.download('stopwords')    
    
    
    def _preprocess_text(text):
        
        corpus=[]
        stem=PorterStemmer()
        lem=WordNetLemmatizer()
        for news in text:
            words=[w for w in word_tokenize(news) if (w not in stop)]

            words=[lem.lemmatize(w) for w in words if len(w)>2]

            corpus.append(words)
        return corpus
    
    corpus=_preprocess_text(text)
    
    dic=gensim.corpora.Dictionary(corpus)
    bow_corpus = [dic.doc2bow(doc) for doc in corpus]
    
    lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = topic_num , 
                                   id2word = dic,                                    
                                   passes = 10,
                                   workers = 2)
    
    return lda_model, bow_corpus, dic

def plot_lda_vis(lda_model, bow_corpus, dic):
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dic)
    pyLDAvis.show(vis)
    return vis


class NLP_ZH_Assistant:
    """
    @method
    
    1. show% : NLP EDA assistant 
    
    2. update% : adjust attr in this class
    
    3. ml% : machine learning algorithm
    """
    
    def __init__(self , df , text , stopwords ,big_category):
        """
        Input:
            df(DataFrame) : data

            text(str): text feature name
                
            stopwords(set) : stopwords for clean text 

            big_category(str) : 最大類別的變數
        
        Example:
            (df , 'text' , set , 'channel')
        
        """
        
        self.df = df
        self.text = text
        
        #Text series
        self.content  = df[text]
        def chinese_word_cut(mytext):
            try:
                word = " ".join(jieba.cut(mytext))
            except:
                word = ''
            return word
        self.big_category = big_category
        corpus = df[text].apply(lambda x: chinese_word_cut(x))
        self.corpus = corpus
        df['cut_text'] = corpus.apply(lambda x:x.split())
        self.df = df
        self.nlp = en_core_web_sm.load()
        self.top_word = None
        self.stopwords = stopwords
        self.words_cooc_matrix = None
#         self.stopwords = set(stopwords.words('english'))
        all_content = []
        for i in list(df[text]):
            all_content.append({'content':str(i)})
        self.all_content = all_content

    
    def show_each_text_length_dist(self):
        self.content.str.len().hist()
        plt.title('The number of characters present in each sentence')
        plt.show()
    
    def show_each_text_used_word_dist(self):
        
        self.df[self.text].str.split().\
        map(lambda x: len(x)).\
        hist()
        plt.title('The number of words appearing in each news headline.')
        plt.show()
    
    def show_average_word_length_in_each_sentence(self): 
        print(' Stopwords are the words that are most commonly used in any language such as “the”,” a”,” an” etc. As these words are probably small in length these words may have caused the above graph to be left-skewed.')
        self.df[self.text].str.split().\
           apply(lambda x : [len(i) for i in x]). \
           map(lambda x: np.mean(x)).hist()
        plt.title('The average word length in each sentence.')
        plt.show()
    
    def show_top_stopwords_bar_plot(self):
        text = self.df[self.text]
        stop= self.stopwords

        
        #Store it into our corpus
        corpus  = self.corpus 
        from collections import defaultdict
        dic=defaultdict(int)
        for word in corpus:
            if word in stop:
                dic[word]+=1
        
        top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
        x,y=zip(*top)
        plt.bar(x,y)
        plt.title('Top_stopwords_bar_plot')
        plt.show()
        
    def show_word_freq_except_to_stopwords(self , first_num):
        """
        Input:
            
            first_num(int): How many first common words you want
        """
        counter=Counter(np.sum(self.df.cut_text))
        stop=self.stopwords
        most=counter.most_common()
        x, y= [], []
        for word,count in most[:first_num]:
            if (word not in stop):
                x.append(word)
                y.append(count)
        plt.title('Word Frequency in our text ')
        sns.barplot(x=y,y=x)
        plt.show()
        
    def show_top_ngram(self, n=None , first_num = 10):
        """
        Input:
            n(int) : N-gram params
            first_num(int) : How many first common words you want
        """
        text = self.text
        df = self.df
        
        corpus = self.corpus
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(df[text])
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) 
                      for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        
        top_n_bigrams= words_freq[:first_num]
        x,y=map(list,zip(*top_n_bigrams))
        sns.barplot(x=y,y=x)
        freq_n_gram = set(sum([x.split() for x in pd.DataFrame(top_n_bigrams)[0]],[]))
        return list(freq_n_gram)
    
    def clean_text_except_chinese(self):
        
        content = self.content
        stop_words = self.stopwords
        punctuations = ["《", "》", "【", "】", "｜", "(",")",  "®", "\n", "？", "@", "#", "?", "！", "!" ,"，" , ',' , '►' ,'\n']


        
        print('移除掉標點符號、數字、英文')
        def remove_english(words):
            new_word = ' '
            for string in words:
                if not re.search(r'[a-zA-Z]', string):
                    new_word += string
                else:
                    new_word += ' '
            return new_word
        for word in stop_words:
        #空白來抽換掉
            content = content.replace(word , ' ')
        
            #移除標點符號
        for punc in punctuations:
            content = content.replace(punc , ' ')
            #移除數字
        new_content = ""
        for word in content:
            if word.isdigit() :
                new_content += ' '
            else:
                new_content += word
        #移除掉網址等英文字
        new_content = remove_english(new_content)

        self.content = new_content
    def create_tidy_text(self , group):
        """
        創建tidy text 架構的列表
        Input
            group(str) : 其次分類類別的變數
         
        """
        df = self.df
        corpus = self.corpus
        big_category = self.big_category
        
        
        tidy_text = df.groupby([big_category , group])['cut_text'].sum().apply(pd.Series).stack().reset_index()

        #更改column名稱
        tidy_text = tidy_text.rename(columns = {0 :'segment'})

        #將長度<=1 的去掉
        long_segment = tidy_text.segment.apply(lambda seg : True if len(seg) > 1 else False )

        tidy_text = tidy_text[long_segment].reset_index(drop = True)
        display(tidy_text.head())
        tidy_text['word_count'] = 1
        group_df = pd.DataFrame(tidy_text.groupby([big_category , 'segment']).sum())
        top_word = pd.DataFrame(group_df['word_count'].groupby(level=0, group_keys=False).nlargest(10))
    
        #轉換為Tidy structure
        top_word = top_word.reset_index()
        self.top_word = top_word
        
    def show_word_freq_ranking(self):
        """詞頻分析"""
        top_word = self.top_word
        big_category = self.big_category
        
        
        different_youtubers = df[big_category].unique()
        youtuber_num = len(different_youtubers)


        drop_str = []
        for youtuber in different_youtubers:
            #這個youtuber的詞頻
            plt.figure(figsize = (6,4))
            this_youtuber_word = top_word[top_word[big_category] == youtuber]
            for seg in this_youtuber_word.segment:
                drop_str.append(seg)
            plt.barh(this_youtuber_word.segment , this_youtuber_word.word_count)
            plt.title(youtuber)
            plt.yticks()
        
    def show_TF_IDF_ranking(self):
        import jieba.analyse
        top_word = self.top_word
        youtubers = top_word.groupby(self.big_category).segment.sum()


        for channel,text in zip(youtubers.to_frame().index , youtubers):
            plt.figure(figsize = (6,4))
            plt.title(channel)
            message = pd.DataFrame(jieba.analyse.extract_tags(text , topK= 10,withWeight=True))
            plt.barh(message[0] , message[1])
    
    def create_occr_matrix(self  ,allowed_words):
         
        #把每個youtuber的前三名取出來
        #準備好我們的文件
        documents = []
        for content in df['cut_text']:
            documents.append(content)
        def create_co_occurences_matrix(allowed_words, documents):
            """
            Input:
                allowed_words:單維列表，也就是希望觀察的詞彙兩兩間關係
                documents:二維列表
            Output:
                value_matrix : 詞與詞的共同出現頻率
            """
            word_to_id = dict(zip(allowed_words, range(len(allowed_words))))
            documents_as_ids = [np.sort([word_to_id[w] for w in doc if w in word_to_id]).astype('uint32') for doc in documents]
            row_ind, col_ind = zip(*itertools.chain(*[[(i, w) for w in doc] for i, doc in enumerate(documents_as_ids)]))
            data = np.ones(len(row_ind), dtype='uint32')  # use unsigned int for better memory utilization
            max_word_id = max(itertools.chain(*documents_as_ids)) + 1
            docs_words_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(documents_as_ids), max_word_id))  # efficient arithmetic operations with CSR * CSR
            words_cooc_matrix = docs_words_matrix.T * docs_words_matrix  # multiplying docs_words_matrix with its transpose matrix would generate the co-occurences matrix
            words_cooc_matrix.setdiag(0)
        #     print(f"words_cooc_matrix:\n{words_cooc_matrix.todense()}")
            return words_cooc_matrix.todense(), word_to_id 
        words_cooc_matrix , word_to_id =  create_co_occurences_matrix(allowed_words , documents)
        words_cooc_matrix =  pd.DataFrame(words_cooc_matrix)
        words_cooc_matrix.index = allowed_words
        words_cooc_matrix.columns = allowed_words
        self.words_cooc_matrix = words_cooc_matrix
        return words_cooc_matrix
    
    def show_co_occurence_structure(self,heat = True,  graph = True):
        """共現矩陣視覺化"""
        if heat:
            plt.figure(figsize = (16,9))
            sns.heatmap(self.words_cooc_matrix, cmap="YlGnBu")
        
        if graph:
            plt.figure(figsize = (16,10))
            #從matrix轉成圖
            G =  nx.from_pandas_adjacency(self.words_cooc_matrix)
            # Graph with Custom nodes:
            nx.draw(G, with_labels=True, node_size=100, node_color="skyblue", node_shape="s", alpha=0.5, linewidths=40)
            plt.show()
        
    def show_word_cloud(self , font_path):
        """
        文字雲視覺化
        """
        top_word = self.top_word
        top_word_dict = dict()
        for segment , _count in zip(top_word.segment , top_word.word_count):
            top_word_dict[segment] = _count 
            
        from wordcloud import WordCloud
        cloud = WordCloud(font_path = font_path ).generate_from_frequencies(top_word_dict)
        plt.figure()
        plt.figure( figsize=(16,8), facecolor='k')
        plt.imshow(cloud,interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0)
        #顯示用
        plt.show()
        
        # 接收語料(dict)與關鍵詞(string)，回傳語料當中關鍵詞出現的句子中，按照頻率排列的動詞表
    def humanities_generate_verb_list(self , word):
#         try:
        raw_text_list = self.all_content
        a = [generate_line_sentence(article['content']) for article in raw_text_list]
        l = [sen for article in a for sen in article if word in sen]
        l = [list(pseg.cut(ll)) for ll in l]
        wordlist = [word for sen in l for (word, pos) in sen if pos.startswith('v')]
        freq_dict = {w:wordlist.count(w) for w in set(wordlist)}
        verb_df = sorted(freq_dict.items(), key=lambda kv: -kv[1])
        temp_df = pd.DataFrame(verb_df)
        s = pd.Series(temp_df[1])
        s.index = temp_df[0]
        display(s.head())
        s = pd.DataFrame(s)
        s.columns = ['count']
        s.query('count > 5').iplot(kind ='bar')
        return sorted(freq_dict.items(), key=lambda kv: -kv[1])
#         except:
#             print('語料庫中不存在此單詞')

    # 接收語料(dict)與關鍵詞(string)，回傳語料中含有該關鍵詞的concordance dataframe
    def humanities_generate_concordance_df(self ,  word):
        try:
            raw_text_list = self.all_content
            a = [generate_line_sentence(article['content']) for article in raw_text_list]
            l = [sen.split(word) for article in a for sen in article if word in sen]
            df = pd.DataFrame({
            "left_context": [s[0] for s in l],
            "keyword": word,
            "right_context": [s[1] for s in l],
            })
            return df
        except:
            print('語料庫中不存在此單詞')
            
   

    def ml_lda_script(self , topic_num , show_lda_vis = False):
        """
        LDA modeling
        """
        text = self.content
        print('Start LDA modeling...')
        lda_model, bow_corpus, dic = train_lda_objects(text , topic_num , self.stopwords)
        word_dict = {};
        for i in range(topic_num):
            #先列出每個主題的前20個關鍵詞
            words = lda_model.show_topic(i, topn = 20);
            word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words];
       
        display( pd.DataFrame(word_dict))
        
        if show_lda_vis:
            print('Dashboard printing...')
            plot_lda_vis(lda_model, bow_corpus, dic)
            

# font_path = '/Users/Dennis/ttf/SimHei 拷貝.ttf'
# tool = NLP_ZH_Assistant(df , 'content' , stopwords = stopwords , big_category='from')
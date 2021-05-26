from django.http import response
from django.shortcuts import HttpResponse, render
from eunjeon import Mecab
from gensim import models
from elasticsearch import Elasticsearch 

client = Elasticsearch('115.85.182.110:9200')
tagger = Mecab()
review_only_model = models.fasttext.load_facebook_model("D:/embedding/word-embeddings/word-embeddings/fasttext/final_review.bin")
stop_tags = ['JKS','JKC','JKG','JKO','JKB','JKV','JKQ','JX','JC',
             'EP','EF','EC','ETN','ETM','XSN','XSV','XSA',
             'SF','SE','SSO','SSC','SC','SY']
stop_words = ['곳','집','식당','장소','음식점']

def get_search_token(sent):
    spl_tokens = sent.split(' ')
    res = []
    for st in spl_tokens:
        pos_st = tagger.pos(st)
        mor_st = tagger.morphs(st)
        while pos_st and ( pos_st[-1][1] in stop_tags or pos_st[-1][0] in stop_words ):
            pos_st = pos_st[:-1]
            mor_st = mor_st[:-1]
        tmp = ''.join(mor_st)
        if len(tmp) > 1:
            res.append(tmp)
    return res

# Create your views here.

# 메인 검색 페이지
def index(request):
    return render(request, 'coogle_search.html')

def get_list(request):
    search = request.GET.get('search_key')
    response = get_info(search)
    res_view_list = info_parser(response)
    return render(request,'res_list_page.html',{'res_view_list':res_view_list,'query':search})

def get_info(search):
    word_list = get_vector_similar_words(get_search_token(search))
    q = make_query(word_list)
    q['highlight'] = {"fields": {"REV_COMMENT": {}}}
    q["_source"] = {"includes": ["RES_ID_NEW","RES_NAME","RES_ADDRESS","REV_COMMENT"]}
    q["size"] = 1000
    response = client.search(
        index='public',
        body=q,
    )
    return response

def get_vector_similar_words(keywords):
    result = []
    for keyword in keywords:
        similar_list = review_only_model.wv.most_similar(keyword, topn=300)
        words = [keyword]
        res = []
        for item in similar_list:
            if keyword not in str(item[0]):
                res.append(item)
        while len(words) < 5 and res:
            tmp = get_search_token(res.pop(0)[0])
            if tmp:
                tmp = tmp[0]
                words.append(tmp)
                tmp_res = []
                for item in res:
                    if tmp not in str(item[0]) and item[1] > 0.7:
                        tmp_res.append(item)
                res = tmp_res
        result.append(words)
    return result

def make_query(word_list):
    main_form = {"query":{"bool":{"must":[]}}}
    for words in word_list:
        and_form = {"bool":{"should":[]}}
        for word in words:
            or_form = {"wildcard":{"REV_COMMENT": ""}}
            or_form['wildcard']['REV_COMMENT'] = "*"+word+"*"
            and_form['bool']['should'].append(or_form)
        main_form['query']['bool']['must'].append(and_form)
    return main_form

def info_parser(info):
    res_list = {}
    for i in range(len(info['hits']['hits'])):
        res_id = info['hits']['hits'][i]['_source']['RES_ID_NEW']
        res_name = info['hits']['hits'][i]['_source']['RES_NAME']
        res_addr = info['hits']['hits'][i]['_source']['RES_ADDRESS']
        rev_comment = info['hits']['hits'][i]['_source']['REV_COMMENT']
        rev_highlight = info['hits']['hits'][i]['highlight']['REV_COMMENT'][0].replace('<br>','.')

        # res_id를 기준으로 같은 식당이 나오는지 검사
        if res_list.get(res_id) == None:
            res_info = {'res_name':res_name, 'res_addr':res_addr, 
                        'rev_comment':[rev_comment],'rev_highlight':[rev_highlight]
            }
            res_list[res_id] = res_info
        else:
            if rev_comment not in res_list[res_id]['rev_comment']:
                res_list[res_id]['rev_comment'].append(rev_comment)
            if rev_highlight not in res_list[res_id]['rev_highlight']:
                res_list[res_id]['rev_highlight'].append(rev_highlight)
    result = []
    for _,res_info in res_list.items():
        result.append((len(res_info['rev_comment']),res_info))
    result.sort(key=lambda x: x[0],reverse=True)
    return [ res_info for _,res_info in result]
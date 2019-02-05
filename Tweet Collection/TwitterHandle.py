# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 23:24:36 2017

@author: Anand Rawat
"""
from re import sub, compile, findall

from datetime import datetime
from time import sleep
from twython import Twython
from Utility import create_set
import os.path
import Properties
from WebScrapperMain import Scrapper
from sys import exc_info


APP_KEY = Properties.APP_KEY  # consumer keys
APP_SECRET = Properties.APP_SECRET  # consumer secret
OAUTH_TOKEN = Properties.OAUTH_TOKEN
OAUTH_TOKEN_SECRET = Properties.OAUTH_TOKEN_SECRET
SEARCH_QUERY = 'WorldCupDraw'
REPLIES = 'Replies_' + SEARCH_QUERY
id_data_set = create_set('dict_'+SEARCH_QUERY+'.txt')
print(id_data_set)
url_data_set = dict()
twitter = Twython(APP_KEY, APP_SECRET,
                  OAUTH_TOKEN, OAUTH_TOKEN_SECRET)


init = True
tweets = ''
scrapper = Scrapper(REPLIES)
mode = str()
if os.path.isfile(SEARCH_QUERY + ".csv"):
    mode = 'a'
else:
    mode = 'w'
print(mode)
count = 0
# sleep(900)
with open(SEARCH_QUERY + ".csv", mode, encoding='utf-8') as mainquery:
    if mode == 'w':
        mainquery.write('id,text,screen_name,retweet_count,favorite_count,friends_count,followers_count,url\n')
    last_id = 10000000000000000000000000000000000000000
    for _ in range(20):
        try:
            for i in range(180):
                if count == 0:
                    last_id -=180
                count = 0
                print('request number: ', i+1)
                q = '#' + SEARCH_QUERY
                if init:
                    tweets = twitter.search(q=q,
                                            count=100,  # not guaranteed; maximum number that can be returned
                                            lang='en',
                                            result_type='recent')
                else:
                    tweets = twitter.search(q=q,
                                            count=100,  # not guaranteed; maximum number that can be returned
                                            lang='en',
                                            max_id=last_id,
                                            result_type='recent')
                for t in tweets['statuses']:
                    # print(t)
                    id = t['id']
                    int_id = int(id)
                    if int_id < last_id:
                        last_id = int_id - 1
                    if id in id_data_set:
                        print('duplicate tweet found', id)
                    else:
                        text = t['text']
                        text = text.replace('\n', ' ')
                        text = text.replace('\r', ' ')
                        # print(text)
                        url_pattern = 'https?:\/\/[a-zA-Z0-9@:%._\+~#=\/]*[ ]*'
                        # url_pattern = 'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)'
                        url_pattern_re = compile(url_pattern)
                        urls = findall(url_pattern_re, text)
                        tweet_url = ''
                        if urls:
                            # print(urls)
                            urls = urls[0].split()
                            if len(urls) > 0:
                                tweet_url = urls[len(urls) - 1].strip()
                                # print(tweet_url)
                        text = sub(url_pattern_re, '', text)
                        text = text.replace(',', ' ')
                        text = text.strip()
                        # print(text,tweet_url)
                        if text == '':
                            print('empty string found')
                        replies_avail = False
                        data = []
                        if tweet_url:
                            if tweet_url not in url_data_set or url_data_set[tweet_url]:
                                scrapper.startscrapping(id, tweet_url)
                                replies_avail = scrapper.repliesfound()
                                url_data_set[tweet_url] = scrapper.repliesfound()
                            else:
                                # print('duplicate url found:',tweet_url,url_data_set[tweet_url])
                                replies_avail = scrapper.repliesfound()
                        if replies_avail:
                            data.append(str(id))
                            data.append(text)
                            data.append(t['user']['screen_name'])
                            data.append(str(t['retweet_count']))
                            data.append(str(t['favorite_count']))
                            data.append(str(t['user']['friends_count']))
                            data.append(str(t['user']['followers_count']))
                            data.append(tweet_url)
                            # print('data',data)
                            line = ','.join(data)
                            line += '\n'
                            mainquery.write(line)
                            count+=1
                            # data['replies'].append(False)
                        # print(replies_avail)
                        # last_id = id
                        id_data_set.add(id)
                # print('last id', last_id)
                init = False
                print('count for request: ', count)
            print('sleeping for 15 mins:',datetime.now())
            sleep(900)
        except :
            t1, v1, trace = exc_info()
            print(t1)
            print(v1)
            print(trace)
            print('sleeping for 5 mins:', datetime.now())
            sleep(300)
print('number of tweets logged:',len(id_data_set))
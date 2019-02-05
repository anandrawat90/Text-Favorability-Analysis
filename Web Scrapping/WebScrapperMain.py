from urllib.request import urlopen as ureq
from bs4 import BeautifulSoup as soup
# from sys import exc_info
from re import sub as resub
from os import path

# -*- coding: UTF-8 -*-


class Scrapper():
    def __init__(self, replies):
        self.replies_title = replies+'.csv'
        mode = ''
        if path.isfile(self.replies_title):
            mode = 'a'
        else:
            mode = 'w'
        with open(self.replies_title, mode, encoding="utf-8") as replies:
            if mode =='w':
                replies.write('in_reply_to,text,emojis_used,retweet_count,like_count\n')
    # def __init__(self, url, id):
    #     self.url = url
    #     self.id = "Replies/"+str(id)+'.csv'
    #     print(self.url,self.id)
    #     self.replies_found = False

    def repliesfound(self):
        return self.replies_found

    def startscrapping(self,id,url):
        self.replies_found = False
        with open(self.replies_title,"a",encoding="utf-8") as replies:
            try:
                uclient = ureq(url)
                tweet_html = uclient.read()
                uclient.close()
                tweet_soup = soup(tweet_html,'html.parser')
                containers = tweet_soup.find_all('div', {'class': 'js-tweet-text-container'})
                if len(containers) > 1:
                    record = False
                    for container in containers:
                        if record:
                            reply = container.p.text.strip()
                            reply = reply.replace('\n',' ')
                            reply = reply.replace('\r','')
                            reply = resub(r'https?:\/\/[a-zA-Z0-9@:%._\+~#=\/]*[ ]*',' ',reply)
                            if not reply.strip():
                                continue
                            emoji_used = ''
                            emojis = container.find_all('img',{'class':'Emoji Emoji--forText'})
                            for emoji in emojis:
                                # reply+="<"+emoji['title']+"-"+emoji['alt']+">"
                                emoji_used += emoji['alt'] + ' '
                            footer = container.parent.find_all('div', {'class': 'stream-item-footer'})[0]
                            retweet_count = footer.find_all('div', {'class': 'ProfileTweet-action--retweet'})[0].find_all('span', {
                                'class': 'ProfileTweet-actionCountForPresentation'})[0].text.strip()
                            liked_count = footer.find_all('div', {'class': 'ProfileTweet-action--favorite'})[0].find_all('span', {
                                'class': 'ProfileTweet-actionCountForPresentation'})[0].text.strip()
                            # print(reply, retweet_count, liked_count)
                            if not retweet_count:
                                retweet_count = 0
                            if not liked_count:
                                liked_count = 0
                            reply = reply.replace(',',' ')
                            reply_data = str(id) + ',' + str(reply) +','+ emoji_used+ ',' + str(retweet_count) + ',' + str(liked_count) + '\n'
                            # print(reply_data)
                            replies.write(reply_data)
                            self.replies_found = True
                        record = True
            except:
                # t1, v1, trace = exc_info()
                # print(t1)
                # print(v1)
                # print(trace)
                print('error occured with: ',url)
                self.replies_found = False

# scrap = Scrapper('https://t.co/36UkqQikKc','932115407783174145')
# scrap.startscrapping()
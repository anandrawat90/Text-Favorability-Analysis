import csv
from os import SEEK_END, path
from urllib.request import urlopen as ureq
from bs4 import BeautifulSoup as soup
from sys import exc_info
from re import sub as resub


def create_set_from_file(file_name):
    output_file = 'dict_' + file_name + '.txt'
    if path.isfile(file_name + ".csv"):
        with open(output_file, 'w', ) as output_dict:
            with open(file_name + '.csv', 'r', encoding='utf-8') as source_file:
                csv_reader = csv.reader(source_file, delimiter=',')
                next(csv_reader)
                for row in csv_reader:
                    output_dict.write(row[0] + ',')
        with open(output_file, 'rb+') as output_dict:
            output_dict.seek(-1, SEEK_END)
            output_dict.truncate()
    else:
        print('No', output_file, 'found!!')


def create_set(file_name):
    ids = set()
    if path.isfile(file_name):
        with open(file_name, newline='\n') as set_file:
            id_list = set_file.readline().split(',')
            for id in id_list:
                ids.add(int(id))
    return ids


def create_replies_set_with_emojies(tweet_file):
    FILE_NAME = 'Replies_' + tweet_file + "_with_Emoji.csv"
    FILE_HEADER = ['in_reply_to', 'text', 'emojies_used', 'retweet_count', 'like_count']
    with open(tweet_file + '.csv', 'r', encoding='utf-8') as tweets:
        with open(FILE_NAME, 'w', encoding='utf-8', newline='') as replies_with_emoji:
            csv_writer = csv.writer(replies_with_emoji)
            csv_writer.writerow(FILE_HEADER)
            csv_reader = csv.reader(tweets)
            next(csv_reader)
            for tweet in csv_reader:
                print('tweet', tweet)
                id = tweet[0]
                url = tweet[7]
                replies = create_replies_with_emojies(id, url)
                print('replies')
                for reply in replies:
                    print(reply)
                    csv_writer.writerow(reply)


def create_replies_with_emojies(id, url):
    replies = list()
    try:
        uclient = ureq(url)
        tweet_html = uclient.read()
        uclient.close()
        tweet_soup = soup(tweet_html, 'html.parser')
        containers = tweet_soup.find_all('div', {'class': 'js-tweet-text-container'})
        if len(containers) > 1:
            record = False
        for container in containers:
            if record:
                reply = container.p.text.strip()
                reply = reply.replace('\n', ' ')
                reply = reply.replace('\r', '')
                reply = resub(r'https?:\/\/[a-zA-Z0-9@:%._\+~#=\/]*[ ]*', ' ', reply)
                emojis_used = ''
                if not reply.strip():
                    continue
                emojis = container.find_all('img', {'class': 'Emoji Emoji--forText'})
                for emoji in emojis:
                    # reply+="<"+emoji['title']+"-"+emoji['alt']+">"
                    emojis_used += " " + emoji['alt']
                footer = container.parent.find_all('div', {'class': 'stream-item-footer'})[0]
                retweet_count = \
                    footer.find_all('div', {'class': 'ProfileTweet-action--retweet'})[0].find_all('span', {
                        'class': 'ProfileTweet-actionCountForPresentation'})[0].text.strip()
                liked_count = \
                    footer.find_all('div', {'class': 'ProfileTweet-action--favorite'})[0].find_all('span', {
                        'class': 'ProfileTweet-actionCountForPresentation'})[0].text.strip()
                # print(reply, retweet_count, liked_count)
                if not retweet_count:
                    retweet_count = 0
                if not liked_count:
                    liked_count = 0
                reply = reply.replace(',', ' ')
                reply_data = [str(id), str(reply), emojis_used, str(retweet_count), str(liked_count)]
                # print(reply_data)
                replies.append(reply_data)
                record = True
    except:
        t1, v1, trace = exc_info()
        print(t1)
        print(v1)
        print(trace)
        print('error occured with: ', url)
        replies.clear()
    return replies


if __name__ == '__main__':
    # create_replies_set_with_emojies('trail')
    create_set_from_file('WorldCupDraw')
    ids = create_set('dict_WorldCupDraw.txt')
    print(len(ids))

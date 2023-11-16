import requests
from bs4 import BeautifulSoup
import pandas as pd


#Scaping the data.
def scrape_data():
    # Gets the contents page, and creates a scaper for that page.
    page = requests.get('https://rajeshjain.com/marketing/').content
    soup = BeautifulSoup(page, 'html.parser')

    # Gets the desired content, which is a list of names of all the articles by the author.
    hrefs = soup.find("div", {"class": "entry-content"}).find_all('a', href=True)
    articles = [tag['href'] for tag in hrefs][:52]

    # Data cleanup for each article.
    def cleanup_content(soup):
        for node in soup.findAll("div")[-5:] + soup.findAll("p")[:1]:
            node.decompose()
        for node in soup(['style', 'script']):
            node.decompose()
        return ' '.join(soup.stripped_strings)


    # Given the article names, iterate over all the articles and get the desired content from each.
    data = {'title': [], 'content': []}
    for c, article in enumerate(articles):
        curr_page = requests.get(article).content
        curr_soup = BeautifulSoup(curr_page, 'html.parser')
        try:
            title = curr_soup.find(
                "header", {"class": "entry-header"}).getText().strip()
            content = curr_soup.find("div", {"class": "entry-content"})
            content = cleanup_content(content)

            data['title'].append(title)
            data['content'].append(content)
        except AttributeError:
            print(curr_soup.getText())

    return data







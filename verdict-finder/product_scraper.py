from bs4 import BeautifulSoup
import requests
from faker import Faker
import json  
import os

def scrape_product_data(url):
    def get_soup_retry(url):
        fake = Faker()
        uag_random = fake.user_agent()

        header = {
            'User-Agent': uag_random,
            'Accept-Language': 'en-US,en;q=0.9'
        }
        isCaptcha = True
        while isCaptcha:
            page = requests.get(url, headers=header)
            assert page.status_code == 200
            soup = BeautifulSoup(page.content, 'html.parser')
            if 'captcha' in str(soup):
                uag_random = fake.user_agent()
                continue
            else:
                return soup

    soup = get_soup_retry(url)

    price = soup.find("span", attrs={"class": 'a-offscreen'}).get_text().strip()
    title = soup.find("span", attrs={"id": 'productTitle'}).text.strip()
    total_review_count = soup.find("div", attrs={"data-hook": 'total-review-count'}).text.strip()

    specs_obj = {}
    specs = soup.find_all("tr", {"class": "a-spacing-small"})
    for u in range(0, len(specs)):
        spanTags = specs[u].find_all("span")
        specs_obj[spanTags[0].text] = spanTags[1].text

    about = []
    specs = soup.find("ul", {"class": "a-unordered-list a-vertical a-spacing-mini"})
    spanTags = specs.find_all("span")
    for u in range(0, len(spanTags)):
        about.append(spanTags[u].text)

    link = []
    for i in soup.findAll("a", {'data-hook': "see-all-reviews-link-foot"}):
        link.append(i['href'])

    def Searchreviews(review_link):
        url = "https://www.amazon.in" + review_link
        return get_soup_retry(url)

    reviews = ""
    for j in range(len(link)):
        for k in range(1, 2):
            new_soup = Searchreviews(link[j] + '&pageNumber=' + str(k))
            for i in new_soup.findAll("span", {'data-hook': "review-body"}):
                reviews = reviews + os.linesep + os.linesep + (i.text.strip())
            if not new_soup.find('li', {'class': 'a-disabled a-last'}):
                pass
            else:
                break

    save_file = open("product_reviews.txt", "w")
    save_file.write("Title: " + title + os.linesep + os.linesep)
    save_file.close()
    file = open("product_reviews.txt", "a")
    file.write("Price: " + price + os.linesep + os.linesep)
    file.write("Specs: " + json.dumps(specs_obj) + os.linesep + os.linesep)
    file.write("About Item: " + json.dumps(about) + os.linesep + os.linesep)
    file.write("Total Review Count: " + total_review_count + os.linesep + os.linesep)
    file.write("Reviews:" + reviews)
    file.close()

    print("Amazon scrape successful", title)
    return title

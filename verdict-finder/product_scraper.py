from bs4 import BeautifulSoup
import requests
from faker import Faker
import json
import os


class AmazonProductScraper:
    def __init__(self):
        self.fake = Faker()
        self.header = {
            'User-Agent': '',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        self.session = requests.Session()
        self.product_data = {}

    def get_soup_retry(self, url):
        is_captcha = True
        while is_captcha:
            self.header['User-Agent'] = self.fake.user_agent()
            response = self.session.get(url, headers=self.header)
            assert response.status_code == 200
            soup = BeautifulSoup(response.content, 'html.parser')
            if 'captcha' in str(soup):
                continue
            else:
                return soup

    def scrape_product_data(self, url):
        soup = self.get_soup_retry(url)

        price = soup.find("span", attrs={"class": 'a-offscreen'}).get_text().strip()
        title = soup.find("span", attrs={"id": 'productTitle'}).text.strip()
        total_review_count = soup.find("div", attrs={"data-hook": 'total-review-count'}).text.strip()

        specs_obj = {}
        specs = soup.find_all("tr", {"class": "a-spacing-small"})
        for u in range(0, len(specs)):
            span_tags = specs[u].find_all("span")
            specs_obj[span_tags[0].text] = span_tags[1].text

        about = []
        specs = soup.find("ul", {"class": "a-unordered-list a-vertical a-spacing-mini"})
        span_tags = specs.find_all("span")
        for u in range(0, len(span_tags)):
            about.append(span_tags[u].text)

        link = [i['href'] for i in soup.findAll("a", {'data-hook': "see-all-reviews-link-foot"})]

        reviews = ""
        for review_link in link:
            url = "https://www.amazon.in" + review_link
            new_soup = self.get_soup_retry(url)
            for i in new_soup.findAll("span", {'data-hook': "review-body"}):
                reviews += os.linesep + os.linesep + i.text.strip()
            if not new_soup.find('li', {'class': 'a-disabled a-last'}):
                break

        with open("product_reviews.txt", "w") as file:
            file.write("Title: " + title + os.linesep + os.linesep)
            file.write("Price: " + price + os.linesep + os.linesep)
            file.write("Specs: " + json.dumps(specs_obj) + os.linesep + os.linesep)
            file.write("About Item: " + json.dumps(about) + os.linesep + os.linesep)
            file.write("Total Review Count: " + total_review_count + os.linesep + os.linesep)
            file.write("Reviews:" + reviews)

        print("Amazon scrape successful:", title)

        # Store the scraped data in the product_data dictionary
        self.product_data['title'] = title
        self.product_data['price'] = price
        self.product_data['specs'] = specs_obj
        self.product_data['about'] = about
        self.product_data['total_review_count'] = total_review_count
        self.product_data['reviews'] = reviews

        return title

    def get_product_data(self):
        # Return the stored product data dictionary
        return self.product_data

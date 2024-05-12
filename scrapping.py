import requests
import os
from uuid import uuid4
from bs4 import BeautifulSoup


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def download_image(url, directory):
    try:
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            random_name = f"{uuid4()}.jpg"
            with open(f"{directory}/{random_name}", 'wb') as img:
                img.write(response.content)
            return random_name
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")


def scrape_page(url, directory, target):
    try:
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            target_divs = soup.find_all('div', class_=target)
            for target_div in target_divs:
                img_tag = target_div.find('img', class_="img-thumbnail")
                if img_tag is None:
                    return

                img_url = img_tag.get('src')
                img_custom_url = f"{img_url}&width=600&height=800"
                img_full_url = f"https://opac.pnj.ac.id/{img_custom_url}"
                img_name = download_image(img_full_url, directory)

                title = target_div.find('a', class_="titleField").text.strip()
                author_divs = target_div.find_all('span', class_='author-name')
                author = ' - '.join([tag.text.strip() for tag in author_divs]).lower()
                
    except Exception as e:
        print(f"Error while scraping page {url}: {e}")


def scrape_pages(base_url, directory, number_of_pages, target):
    try:
        books = []
        for page_number in range(1, number_of_pages + 1):
            url = f"{base_url}&page={page_number}"
            book = scrape_page(url, directory, target)
            books.append(book)

        return books

    except Exception as e:
        print(f"Error while scraping pages: {e}")


def main():
    url = "https://opac.pnj.ac.id/index.php?keywords=&search=search"
    directory = "train_images/"
    create_directory(directory)
    number_of_pages = 2
    target = "item biblioRecord"

    result = scrape_pages(url, directory, number_of_pages, target)
    print(result)


if __name__ == "__main__":
    main()

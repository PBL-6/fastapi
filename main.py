from fastapi import File, UploadFile, FastAPI, HTTPException, status, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
from uuid import uuid4
from sqlmodel import Field, SQLModel, create_engine, Session, select
from databases import Database
from bs4 import BeautifulSoup
from typing import Optional
from sqlalchemy import cast, Date
import re
import os
import requests
import time
import search1
import pytesseract
import datetime


class Book(SQLModel, table=True):
    __tablename__ = "books"
    id: int = Field(default=None, primary_key=True)
    title: str
    author: str
    image: str
    location: Optional[str]
    is_available: bool = False
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


class SearchBook(SQLModel, table=True):
    __tablename__ = "search_books"
    id: int = Field(default=None, primary_key=True)
    query_image_name: str
    result_image_name_1: Optional[str]
    result_image_name_2: Optional[str]
    result_image_name_3: Optional[str]
    result_image_name_4: Optional[str]
    type: int = Field(default=None)
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


IMAGEDIR = "query_images/"
DATABASE_URL = "mysql://root:@localhost:3306/pbl-6"

database = Database(DATABASE_URL)
engine = create_engine(DATABASE_URL)


def create_database():
    SQLModel.metadata.create_all(engine)


app = FastAPI()
app.mount("/train_images", StaticFiles(directory="train_images"), name="train_images")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await database.connect()
    # create_database()


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


# scrape
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


def scrape_page(url, directory, target, width=600, height=800):
    try:
        time.sleep(1)
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            target_divs = soup.find_all('div', class_=target)
            for target_div in target_divs:

                img_tag = target_div.find('img', class_="img-thumbnail")

                if img_tag is None:
                    return

                img_url = img_tag.get('src')
                img_custom_url = f"{img_url}&width={width}&height={height}"
                img_full_url = f"https://opac.pnj.ac.id/{img_custom_url}"
                img_name = download_image(img_full_url, directory)

                title = target_div.find('a', class_="titleField").text.strip()
                author_divs = target_div.find_all('span', class_='author-name')
                author = ' - '.join([tag.text.strip() for tag in author_divs]).lower()

                with Session(engine) as session:
                    book = Book(title=title, author=author, image=img_name)
                    session.add(book)
                    session.commit()
                    session.refresh(book)

    except Exception as e:
        print(f"Error while scraping page {url}: {e}")


def scrape_pages(base_url, directory, number_of_pages, target, width, height):
    try:
        for page_number in range(1, number_of_pages + 1):
            url = f"{base_url}&page={page_number}"
            scrape_page(url, directory, target, width, height)

    except Exception as e:
        print(f"Error while scraping pages: {e}")


def scrapping(pages, width, height):
    url = "https://opac.pnj.ac.id/index.php?keywords=&search=search"
    directory = "train_images/"
    create_directory(directory)
    number_of_pages = pages
    target = "item biblioRecord"

    scrape_pages(url, directory, number_of_pages, target, width, height)


def image_to_string(image_name):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    image_title = pytesseract.image_to_string(f"query_images/{image_name}", lang='ind')
    image_title = image_title.replace('\n', ' ')
    image_title = image_title.replace('  ', ' ')
    return image_title


async def resize_image(image):
    image = Image.open(BytesIO(await image.read()))
    if image.mode == "RGBA" or image.mode == "P":
        image = image.convert("RGB")

    width = 600
    height = 800

    resized_image = image.resize((width, height), Image.Resampling.LANCZOS)
    image_name = f"{uuid4()}.jpg"

    resized_image.save(f"{IMAGEDIR}{image_name}")

    return image_name


# api
@app.post("/admin/scrape")
async def scrapping_books(pages: int = Form(...), image_width: int = Form(...), image_height: int = Form()):
    scrapping(pages, image_width, image_height)
    search1.calculate_train_images_histogram()
    return {
        "status": "success",
        "message": "scrapping success.",
        "data": ""
    }


@app.post("/admin/books")
async def create_books(title: str = Form(...), author: str = Form(...), image: UploadFile = File(...)):
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="unsupported image type.")

    image_name = await resize_image(image)

    with Session(engine) as session:
        book = Book(title=title, author=author, image=image_name)
        session.add(book)
        session.commit()
        session.refresh(book)

    search1.calculate_train_images_histogram()

    return {
        "status": "success",
        "message": "upload image successful.",
        "data": book
    }


@app.post("/user/books/search")
async def search_books(image: UploadFile = File(...)):
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="unsupported image type.")

    image_name = await resize_image(image)

    books = search1.main(image_name)
    # image_names = [(item[0], item[1].split('/')[-1]) for item in books]
    results = books if books is not None else image_to_string(image_name)

    return {
        "status": "success",
        "message": "search book success.",
        "data": results
    }

    # books_data = []
    #
    # for res in results:
    #     img_name = re.search(r'[^/]+$', res[1]).group(0)
    #     with Session(engine) as session:
    #         statement = select(Book).where(Book.image == img_name)
    #         result = session.exec(statement).first()
    #         result.image = f"127.0.0.1:8000/{res[1]}"
    #         books_data.append(result)
    #
    # img_name_1 = books_data[0].image if len(results) >= 1 else None
    # img_name_2 = books_data[1].image if len(results) >= 2 else None
    # img_name_3 = books_data[2].image if len(results) >= 3 else None
    # img_name_4 = books_data[3].image if len(results) >= 4 else None
    #
    # with Session(engine) as session:
    #     search_book = SearchBook(query_image_name=image_name, result_image_name_1=img_name_1,
    #                              result_image_name_2=img_name_2, result_image_name_3=img_name_3,
    #                              result_image_name_4=img_name_4)
    #     session.add(search_book)
    #     session.commit()
    #     session.refresh(search_book)
    #
    # return {
    #     "status": "success",
    #     "message": "search book success.",
    #     "data": {
    #         "books": books_data
    #     }
    # }


@app.get("/user/books/search/{search_type}")
async def get_search_books(search_type: int):
    search_type_code = [1, 2]

    if search_type in search_type_code:

        with Session(engine) as session:
            statement = (select(SearchBook).where(SearchBook.type == search_type)
                         .where(cast(SearchBook.created_at, Date) == datetime.date.today()))
            results = session.exec(statement)
            books_data = [result.dict() for result in results]

        return {
            "status": "success",
            "message": "search book success.",
            "data": {
                "books": books_data
            }
        }
    else:
        return {
            "status": "error",
            "message": "search type not available.",
            "data": ""
        }


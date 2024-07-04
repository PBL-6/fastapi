from fastapi import File, UploadFile, FastAPI, HTTPException, status, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
from uuid import uuid4
from sqlmodel import Field, SQLModel, create_engine, Session
from databases import Database
from bs4 import BeautifulSoup
from typing import Optional
import os
import requests
import search1
import datetime
import translators as ts
from joblib import load
from urllib3 import disable_warnings
from dotenv import load_dotenv


class Book(SQLModel, table=True):
    __tablename__ = "books"
    id: int = Field(default=None, primary_key=True)
    title: str
    author: str
    image: str
    category: Optional[str]
    is_available: bool = False
    published_at: datetime.date = Field(default_factory=datetime.date.today)
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


class SearchBook(SQLModel, table=True):
    __tablename__ = "search_books"
    id: int = Field(default=None, primary_key=True)
    query_image: str
    book_1_match: Optional[int]
    book_1_image: Optional[str]
    book_2_match: Optional[int]
    book_2_image: Optional[str]
    book_3_match: Optional[int]
    book_3_image: Optional[str]
    book_4_match: Optional[int]
    book_4_image: Optional[str]
    response_time: float
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


load_dotenv()
IMAGEDIR = "query_images/"
DATABASE_URL = os.environ.get("DATABASE_URL")

database = Database(DATABASE_URL)
engine = create_engine(DATABASE_URL)
disable_warnings()


def create_database():
    SQLModel.metadata.create_all(engine)


app = FastAPI()
app.mount("/train_images", StaticFiles(directory="train_images"), name="train_images")
app.mount("/query_images", StaticFiles(directory="query_images"), name="query_images")

global label
global vectorizer
global model

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global label
    global vectorizer
    global model

    await database.connect()
    label = load('label.pkl')
    vectorizer = load('vectorizer.pkl')
    model = load('model.pkl')
    create_database()


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
        print(f"Error while downloading image from {url}: {e}")


def classify(title):
    try:
        text = [ts.translate_text(query_text=title, translator='google', from_language='id', to_language='en')]
        s = (vectorizer.transform(text))
        d = (model.predict(s))
        return label.inverse_transform(d)[0]
    except Exception as e:
        print(f"Error while classifying category: {e}")


def scrape_page(url, directory, target, width=600, height=800):
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
                img_custom_url = f"{img_url}&width={width}&height={height}"
                img_full_url = f"https://opac.pnj.ac.id/{img_custom_url}"
                img_name = download_image(img_full_url, directory)

                title = target_div.find('a', class_="titleField").text.strip().title()
                author_divs = target_div.find_all('span', class_='author-name')
                author = ' - '.join([tag.text.strip().title() for tag in author_divs])

                category = classify(title)

                with Session(engine) as session:
                    book = Book(title=title, author=author, image=img_name, category=category)
                    session.add(book)
                    session.commit()
                    session.refresh(book)

    except Exception as e:
        print(f"Error while scraping page {url}: {e}")


def scrape_pages(base_url, directory, number_of_pages, target, width, height):
    try:
        for page_number in range(1, number_of_pages + 1):
            print(f"scrapping page: {page_number}")
            url = f"{base_url}&page={page_number}"
            scrape_page(url, directory, target, width, height)

    except Exception as e:
        print(f"Error while scraping pages: {e}")


def scrapping(pages, width, height):
    try:
        url = "https://opac.pnj.ac.id/index.php?keywords=&search=search"
        directory = "train_images/"
        create_directory(directory)
        number_of_pages = pages
        target = "item biblioRecord"

        scrape_pages(url, directory, number_of_pages, target, width, height)

    except Exception as e:
        print(f"Error while scraping pages: {e}")


async def resize_image(image, location):
    try:
        image = Image.open(BytesIO(await image.read()))
        image = image.convert("RGB")

        width = 600
        height = 800

        resized_image = image.resize((width, height), Image.Resampling.LANCZOS)
        image_name = f"{uuid4()}.jpg"

        resized_image.save(f"{location}/{image_name}")

        return image_name

    except Exception as e:
        print(f"Error while resizing image: {e}")


# api
@app.post("/admin/books/scrape")
async def scrapping_books(pages: int = Form(...), image_width: int = Form(...), image_height: int = Form()):
    try:
        scrapping(pages, image_width, image_height)
        search1.calculate_train_images_histogram()

        return {
            "status": "success",
            "message": "scrapping success.",
            "data": None
        }

    except Exception as e:
        print(f"Error while resizing image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/admin/books")
async def create_books(title: str = Form(...), author: str = Form(...), category: str = Form(...),
                       published_at: str = Form(...), available: bool = Form(...), image: UploadFile = File(...)):
    try:
        if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Unsupported image type")

        image_name = await resize_image(image, 'train_images')

        with Session(engine) as session:
            book = Book(title=title, author=author, category=category, is_available=available,
                        published_at=published_at, image=image_name)
            session.add(book)
            session.commit()
            session.refresh(book)

        search1.calculate_train_images_histogram()

        return {
            "status": "success",
            "message": "upload image successful.",
            "data": book
        }

    except Exception as e:
        print(f"Error while creating book: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.put("/admin/books/{book_id}")
async def update_book(book_id: int, title: str = Form(...), author: str = Form(...), category: str = Form(...),
                      published_at: str = Form(...), available: bool = Form(...), image: UploadFile = File(None)):
    try:
        with Session(engine) as session:

            book = session.query(Book).filter(Book.id == book_id).first()

            if not book:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found.")

            book.title = title
            book.author = author
            book.category = category
            book.is_available = available
            book.published_at = published_at

            if image:
                if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
                    raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                                        detail="Unsupported image type.")

                image_name = await resize_image(image, 'train_images')
                book.image = image_name

                search1.calculate_train_images_histogram()

            session.commit()
            session.refresh(book)

        return {
            "status": "success",
            "message": "book updated successfully.",
            "data": book
        }

    except Exception as e:
        print(f"Error while updating book: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/admin/books/{book_id}")
async def delete_book(book_id: int):
    try:
        with Session(engine) as session:

            book = session.query(Book).filter(Book.id == book_id).first()

            if not book:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found.")

            if book.image:
                image_path = os.path.join('train_images/', book.image)
                if os.path.exists(image_path):
                    os.remove(image_path)

            session.delete(book)
            session.commit()

            search1.calculate_train_images_histogram()

        return {
            "status": "success",
            "message": "book deleted successfully.",
            "data": None
        }

    except Exception as e:
        print(f"Error while deleting book: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/admin/books/category")
async def classify_books(title: str = Form(...)):
    try:

        category = classify(title)
        return {
            "status": "success",
            "message": "category classification successful",
            "data": category
        }

    except Exception as e:
        print(f"Error while classifying book category: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/user/books/search")
async def search_books(image_name: str = Form(...)):
    try:
        books = search1.main(image_name)

        if books is None:
            return {
                "status": "failed",
                "message": "search book failed.",
                "data": None
            }
        else:
            return {
                "status": "success",
                "message": "search book success.",
                "data": books
            }

    except Exception as e:
        print(f"Error while classifying book category: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/user/books/upload")
async def search_books(image: UploadFile = File(...)):
    try:

        if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="unsupported image type.")

        image_name = await resize_image(image, 'query_images')

        return {
            "status": "success",
            "message": "upload book cover success.",
            "data": image_name
        }

    except Exception as e:
        print(f"Error while uploading book cover: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

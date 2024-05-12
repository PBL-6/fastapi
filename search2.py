import pytesseract
import time


def main(image_name="3cad62e8-dffa-4497-92f9-631260d7f5ae.jpg"):
    start_time = time.time()
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    text = pytesseract.image_to_string(f'query_images/{image_name}', lang='ind')
    text = text.replace('\n', ' ')
    text = text.replace('  ', ' ')
    print(text)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    # return text


if __name__ == "__main__":
    main()

import easyocr

reader = easyocr.Reader(['en'])

def read_license_plate(image):
    results = reader.readtext(image)
    for (bbox, text, prob) in results:
        if len(text) >= 4 and prob > 0.5:
            return text.upper()
    return "N/A"

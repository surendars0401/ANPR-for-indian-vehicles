import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
harcascade = "model/indian_license_plate.xml"

def preprocess_image_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

def extract_text_from_image(img):
    processed_img = preprocess_image_for_ocr(img)
    text = pytesseract.image_to_string(processed_img)
    return text.strip()

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    cap.set(3, 640)
    cap.set(4, 480)
    min_area = 500
    plate_detected_in_last_frame = False

    while True:
        success, img = cap.read()
        if not success:
            break

        plate_cascade = cv2.CascadeClassifier(harcascade)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 10)
        plate_detected_in_this_frame = False

        for (x, y, w, h) in plates:
            area = w * h
            if area > min_area:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                plate_roi = img[y:y+h, x:x+w]
                text = extract_text_from_image(plate_roi)
                cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Number Plate ROI", plate_roi)
                plate_detected_in_this_frame = True
                break

        if not plate_detected_in_this_frame and plate_detected_in_last_frame:
            cv2.destroyWindow("Number Plate ROI")

        plate_detected_in_last_frame = plate_detected_in_this_frame

        cv2.imshow("Result", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

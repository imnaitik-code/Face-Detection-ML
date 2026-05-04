from pathlib import Path

import cv2
import face_recognition


KNOWN_IMAGE = Path("images/Elon Musk/weffsd.jpg")
TEST_IMAGE = Path("images/Jeff Bezos/wefwf.jpg")


def load_face(image_path):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    rgb_image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(rgb_image)

    if not face_locations:
        raise ValueError(f"No face found in image: {image_path}")

    encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
    display_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return display_image, face_locations[0], encoding


try:
    imgElon, faceLoc, encodeElon = load_face(KNOWN_IMAGE)
    imgTest, faceLocTest, encodeTest = load_face(TEST_IMAGE)
except (FileNotFoundError, ValueError) as error:
    print(error)
    raise SystemExit(1)

cv2.rectangle(
    imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2
)
cv2.rectangle(
    imgTest,
    (faceLocTest[3], faceLocTest[0]),
    (faceLocTest[1], faceLocTest[2]),
    (255, 0, 255),
    2,
)

results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
print(results, faceDis)
cv2.putText(
    imgTest,
    f"{results} {round(faceDis[0], 2)}",
    (50, 50),
    cv2.FONT_HERSHEY_COMPLEX,
    1,
    (0, 0, 255),
    2,
)

cv2.imshow("Elon Musk", imgElon)
cv2.imshow("Elon Test", imgTest)
cv2.waitKey(0)
cv2.destroyAllWindows()

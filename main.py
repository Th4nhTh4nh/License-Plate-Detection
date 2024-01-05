import os
import cv2
import pytesseract
from flask import Flask, render_template, send_from_directory, url_for
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from wtforms import SubmitField
from flask_wtf.file import FileAllowed, FileRequired, FileField
from functions import (
    DetectPlate,
    PlatePreprecess,
    CharacterSegment,
    CharactersClassification,
)

app = Flask(__name__)
app.config["SECRET_KEY"] = "fajfklajljkag"
CONFIG = r"--oem 3 --psm 6"
app.config["UPLOADED_PHOTOS_DEST"] = "uploads"
app.config["PLATES_DEST"] = "plates"

photos = UploadSet("photos", IMAGES)
configure_uploads(app, photos)


class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, "Only images"),
            FileRequired("Not empty"),
        ]
    )
    submit = SubmitField("Upload")


@app.route("/uploads/<filename>")
def get_file(filename):
    return send_from_directory(app.config["UPLOADED_PHOTOS_DEST"], filename)


@app.route("/plates/<filename>")
def get_file2(filename):
    return send_from_directory(app.config["PLATES_DEST"], filename)


@app.route("/", methods=["GET", "POST"])
def upload_image():
    text = ""
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for("get_file", filename=filename)
        file_url = "." + file_url
        print(file_url)
        img, plate = DetectPlate(file_url)

        plate_filename = "plate.jpg"
        cv2.imwrite(os.path.join(app.config["PLATES_DEST"], plate_filename), img)
        plate_url = url_for("get_file2", filename=plate_filename)
        plate_url = "." + plate_url

        plate_processed = PlatePreprecess(plate)
        CharacterSegment(plate_processed)
        text = CharactersClassification()
        # text = pytesseract.image_to_string(plate_processed, config=CONFIG)

    else:
        file_url = None
        plate_url = None
    return render_template(
        "index.html", form=form, file_url=file_url, plate_url=plate_url, text=text
    )


if __name__ == "__main__":
    app.run(debug=True)

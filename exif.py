from PIL import Image
from PIL.ExifTags import TAGS

def get_exif_data(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()

    if exif_data is not None:
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            print(f"{tag}: {value}")
    else:
        print("No EXIF data found.")

# Example usage
print('\nPicture from Facebook')
get_exif_data('C:/Users/lenovo/OneDrive/Documents/PROJECTS/ALT/Real-Estate-Image-Matching/pictures/Facebook.jpg')
print('\nPicture from FB Messenger')
get_exif_data('C:/Users/lenovo/OneDrive/Documents/PROJECTS/ALT/Real-Estate-Image-Matching/pictures/FB Messenger.jpg')
print('\nPicture from Google')
get_exif_data('C:/Users/lenovo/OneDrive/Documents/PROJECTS/ALT/Real-Estate-Image-Matching/pictures/Google.jpg')
print('\nPicture from Camera')
get_exif_data('C:/Users/lenovo/OneDrive/Documents/PROJECTS/ALT/Real-Estate-Image-Matching/pictures/Camera.jpg')

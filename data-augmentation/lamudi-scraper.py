import requests, os, csv, time
from bs4 import BeautifulSoup
from unicodedata import normalize

baseurl = "https://www.lamudi.com.ph/buy/"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
}

# property_links = []
# num_pages = 35
timeout_value = 10
# for x in range(1,num_pages):
#     r = requests.get(f'https://www.lamudi.com.ph/buy/?page={x}, timeout={timeout_value}')
#     soup = BeautifulSoup(r.content, 'lxml')
#     property_rows = soup.find_all('div', class_='ListingCell-row')
#     # Iterate over each property row and extract links
#     for property_row in property_rows:
#         links = property_row.find_all('a', class_='ListingCell-ListingLink')  # Find all anchor tags within the row
#         for link in links:
#             href = link.get('href')  # Get the 'href' attribute of the link
#             if href:  # Check if the link has an 'href' attribute
#                 property_links.append(href)
#                 print(href)
#     print(len(property_links))
#     # Create a folder to store the images (if it doesn't exist)
#     folder_path = "dataset"
#     os.makedirs(folder_path, exist_ok=True)
#     file_path = os.path.join(folder_path, "property_links.csv")
#     # Create a CSV file
#     with open(file_path, "w", newline="") as csvfile:
#         csv_writer = csv.writer(csvfile)
#         csv_writer.writerow(["Property Links"])  # Write the header row
#         # Write each property link as a separate row
#         for link in property_links:
#             csv_writer.writerow([link])

#     print(f"30 Property Links of page {x} written to property_links.csv")


folder_path = "dataset"
os.makedirs(folder_path, exist_ok=True)
file_path = os.path.join(folder_path, "property_links.csv")
with open(file_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Skip the header row
    property_links = [row[0] for row in reader]  # Extract the first element (link) from each row

print(property_links)

data = []
start_index = 738  # Starting index
end_index = 1000    # Ending index (exclusive)
for index in range(start_index, end_index+1):
    # testlink = "https://www.lamudi.com.ph/projects/one-rockwell/1-bedroom-loft-condo-for-sale-in-one-rockwell-maka-17164415131/"
    # testlink = property_links[1]
    property_link = property_links[index - 1] 
    r = requests.get(property_link, headers = headers, timeout = 10)
    soup = BeautifulSoup(r.content, 'lxml')
    ID = index

    title = soup.find('h1', class_='Title-pdp-title')
    if title:
        title = title.text.strip()
        print(title)
    else:
        title = "NA" 

    address = soup.find('h3', class_='Title-pdp-address')
    if address:
        address = address.text.strip()
        print(address)
    else:
        address = "NA"

    price = soup.find('span', class_='FirstPrice')
    print(price)
    if price:
        price = price.text.strip()
        price = int(price.replace("₱", "").replace(",", ""))
        print(price)
    else:
        price = "NA"

    attributes = soup.find('div', class_='Title-pdp-attributes')
    if attributes:
        attributes = attributes.text.strip().split()
        if attributes:
            bedrooms = attributes[0] if len(attributes) >= 1 else "NA"
            bathrooms = attributes[1] if len(attributes) >= 2 else "NA"
            floor_area = attributes[2] if len(attributes) >= 3 else "NA"
            print(attributes)
        else:
            bedrooms = bathrooms = floor_area = "NA"
    else:
        bedrooms = bathrooms = floor_area = "NA"

    description = soup.find('div', class_='ViewMore-text-description')
    if description:
        description = description.text.strip()
        print(description)
    else:
        description = "NA"


    amenities = soup.find_all('span', class_='listing-amenities-name')
    if amenities:
        amenities_list = [amenity.text.strip() for amenity in amenities]
        amenities_string = ", ".join(amenities_list)
        print(amenities_string)
    else:
        amenities_string = "NA"
    
    image_urls = soup.find_all('img', class_='jsGalleryMainImage', src=True)

    if image_urls:
        image_urls_string = ", ".join([url['src'] for url in image_urls])
        folder_path = "dataset/authentic_images"
        os.makedirs(folder_path, exist_ok=True)
        for num_img, image_url in enumerate(image_urls, start=1):
            image_url = image_url['src']
            print(image_url)
            filename = f'A{index}.{num_img}.jpg'
            print(filename)
            # Construct the full path for the downloaded image
            file_path = os.path.join(folder_path, filename)
            # Download the image
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)

                print(f"Image downloaded and saved as {filename}")
            else:
                print(f"Failed to download image. Status code: {response.status_code}")
        
    data.append([ID, title, price, address, bedrooms, bathrooms, floor_area, description, amenities_string, image_urls_string])
    with open('dataset/lamudi_data.csv', 'a', encoding='utf-8', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([ID, title, price, address, bedrooms, bathrooms, floor_area, description, amenities_string, image_urls_string])
    print(data)

# with open('dataset/lamudi_data.csv', 'a', encoding='utf-8', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#     # csv_writer.writerow(['ID', 'Title', 'Price', 'Address', 'Num_of_Bedrooms', 'Num_of_Bathrooms', 'Floor_Area', 'Description', 'List_of_Amenities', 'Image URLs'])
#     csv_writer.writerows(data)


# testlink = "https://www.lamudi.com.ph/projects/one-rockwell/1-bedroom-loft-condo-for-sale-in-one-rockwell-maka-17164415131/"
# r = requests.get(testlink, headers = headers)
# soup = BeautifulSoup(r.content, 'lxml')
# image_urls = soup.find_all('img', class_='jsGalleryMainImage', src=True)
# # Create a folder to store the images (if it doesn't exist)
# folder_path = "dataset/test_images"
# os.makedirs(folder_path, exist_ok=True)
# for num_img, image_url in enumerate(image_urls, start=1):
#     image_url = image_url['src']
#     print(image_url)
#     filename = f'{num_img}.jpg'
#     print(filename)
#     # Construct the full path for the downloaded image
#     file_path = os.path.join(folder_path, filename)
#     # Download the image
#     response = requests.get(image_url, stream=True)
#     if response.status_code == 200:
#         with open(file_path, 'wb') as f:
#             for chunk in response.iter_content(1024):
#                 f.write(chunk)

#         print(f"Image downloaded and saved as {filename}")
#     else:
#         print(f"Failed to download image. Status code: {response.status_code}")



# print(soup.find('div', class_='listing-details').text.strip().split())
# labels = soup.find_all('div', class_='listing-details-label')
# for label in labels:
#     print(label.text.strip())
# values = soup.find_all('div', class_='last')
# for value in values:
#     print(value.text.strip())

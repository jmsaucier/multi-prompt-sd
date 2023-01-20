
import requests
import base64
from io import BytesIO
from PIL import Image

model_inputs = {'prompt': '''
                Star Wars scene artificial intelligence 
                a tricolor rabbit wearing golden Jedi knight cape holding a blue lightsabe in it\'s left paw, showing R2D2 and 3PO in background, artstation trends, 
                concept art, highly detailed, intricate, sharp focus, digital art, 8 k
                '''}

res = requests.post('http://localhost:9102/', json = model_inputs)
result = res.json()
encoding = result['encoding']
print(str(encoding))

image_byte_string = result["image_base64"]

image_encoded = image_byte_string.encode('utf-8')
image_bytes = BytesIO(base64.b64decode(image_encoded))
image = Image.open(image_bytes)
image.save("output.jpg")
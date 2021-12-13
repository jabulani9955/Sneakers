import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from operator import itemgetter

train_generator_classes = {
            'Adidas_Originals_Sl80': 0,
            'Adidas_Stan_Smith': 1,
            'New_Balance_237': 2,
            'New_Balance_997': 3,
            'Nike_Air_Force_1_Low': 4,
            'Nike_Air_Huarache': 5,
            'Nike_Air_Max_90': 6,
            'Nike_Air_Max_95': 7,
            'Nike_Blazer': 8,
            'PUMA_Suede': 9,
            'Reebok_Instapump_Fury': 10
        }
        
model = load_model('/home/jabulani/Final_Project/bot/api/checkpoint_V2')


img = image.load_img('/home/jabulani/Final_Project/data/images/output/photo_132032976_177.jpg', target_size=(224, 224))
x = image.img_to_array(img).astype('float32') / 255
x = np.expand_dims(x, axis=0)
preds = model.predict(x)
sorted_preds = list(reversed(np.argsort(preds, axis=1)[:,-4:][0]))
labels = list(train_generator_classes.keys())
selector = itemgetter(*sorted_preds)

print(selector(labels))





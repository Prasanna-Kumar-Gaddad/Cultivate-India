from flask import Flask, render_template, request, Markup ,redirect
import numpy as np
import pandas as pd
import requests
import pickle
import os
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
# =========================================================================================

app = Flask(__name__)


# CROP

# render home page
@ app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('home.html', title=title)

# =========================================================================================

crop_recommendation_model_path = 'model/crop_prediction/RF_pkl.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Harvestify - Crop Recommendation'
    return render_template('crop.html', title=title)


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = "9d7cde1f6d07ec55650544be1631307e"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")
        
        try: 
            if weather_fetch(city) != None:
               temperature, humidity = weather_fetch(city)
               data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
               my_prediction = crop_recommendation_model.predict(data)
               final_prediction = my_prediction[0]
               return render_template('crop-result.html', prediction=final_prediction, title=title)
        except:
            return render_template('crop.html', prediction=final_prediction, title=title)
            

        

# =========================================================================================
@ app.route('/fertilizer')
def fertilizer_recommendation():
    return render_template('fertilizer.html')


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/3).preprocessed data/FertilizerData.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page

# ===============================================================================================


json_file = open('model/disease classification/model.json' , 'r')
loaded_model = json_file.read()

json_file.close()

model = model_from_json(loaded_model)
model.load_weights('model/disease prediction/model.h5')


upload_folder = 'static/uploads'

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    if request.method == 'POST':
        img = request.files['file']
        if img:
            
            image_loc = os.path.join(upload_folder, img.filename)
            
            img.save(image_loc)
            
            img = image.load_img(image_loc, target_size=(224, 224))
            x = image.img_to_array(img)
            x = x/255
            img_data = np.expand_dims(x, axis=0)
            a = np.argmax(model.predict(img_data), axis=1)
    
            prediction = Markup(str(disease_dic[a[0]]))
        return render_template('disease-result.html', prediction=prediction)
    return render_template('dise.html')

# ===============================================================================================
if __name__ == '__main__':
    app.run()

import flask
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
app=flask.Flask(__name__,template_folder='templates')
with open(f'models/knn_regressor_save.sav','rb') as file:
    knn_regressor=joblib.load(file)
with open(f'models/scaler_save.sav','rb') as file2:
    sc=joblib.load(file2)
@app.route('/',methods=['GET', 'POST'])
def main():
    if flask.request.method=='GET':
        return(flask.render_template('main.html'))
    if flask.request.method=='POST':
        apparentTemperatureMax=flask.request.form['apparentTemperatureMax']
        apparentTemperatureMin=flask.request.form['apparentTemperatureMin']
        cloudCover=flask.request.form['cloudCover']
        dewPoint=flask.request.form['dewPoint']
        humidity=flask.request.form['humidity']
        precipIntensity=flask.request.form['precipIntensity']
        precipIntensityMax=flask.request.form['precipIntensityMax']
        precipProbability=flask.request.form['precipProbability']
        precipAccumulation=flask.request.form['precipAccumulation']
        precipTypeIsRain=flask.request.form['precipTypeIsRain']
        precipTypeIsSnow=flask.request.form['precipTypeIsSnow']
        pressure=flask.request.form['pressure']
        temperatureMax=flask.request.form['temperatureMax']
        temperatureMin=flask.request.form['temperatureMin']
        visibility=flask.request.form['visibility']
        windBearing=flask.request.form['windBearing']
        windSpeed=flask.request.form['windSpeed']
        NDVI=flask.request.form['NDVI']
        DayInSeason=flask.request.form['DayInSeason']
        dataToPredict=[[float(apparentTemperatureMax), float(apparentTemperatureMin), float(cloudCover), float(dewPoint), float(humidity), float(precipIntensity),float(precipIntensityMax),float(precipProbability), float(precipAccumulation), float(precipTypeIsRain),float(precipTypeIsSnow), float(pressure), float(temperatureMax),
        float(temperatureMin), float(visibility), float(windBearing), float(windSpeed), float(NDVI), float(DayInSeason)]]
        data_temp=pd.DataFrame(dataToPredict)
        data_removed_cols=data_temp.iloc[:,9:11]
        data_temp=data_temp.drop(data_temp.columns[[9,10]],axis=1)
        data_temp=sc.transform(data_temp)
        data_temp=pd.DataFrame(data_temp)
        scaled_data=pd.concat([data_temp,data_removed_cols],axis=1)
        scaled_data.columns=range(scaled_data.shape[1])
        results=knn_regressor.predict(scaled_data)
        return(flask.render_template('main.html',inputs={'apparentTemperatureMax':apparentTemperatureMax,'apparentTemperatureMin':apparentTemperatureMin,'cloudCover':cloudCover,'dewPoint':dewPoint,'humidity':humidity,'precipIntensity':precipIntensity,'precipIntensityMax':precipIntensityMax,'precipProbability':precipProbability,'precipAccumulation':precipAccumulation,'precipTypeIsRain':precipTypeIsRain,'precipTypeIsSnow':precipTypeIsSnow,'pressure':pressure,'temperatureMax':temperatureMax,'temperatureMin':temperatureMin,'visibility':visibility,'windBearing':windBearing,'windSpeed':windSpeed,'NDVI':NDVI,'DayInSeason':DayInSeason},result=results))
        
if __name__=='__main__':
    app.run(host='0.0.0.0')
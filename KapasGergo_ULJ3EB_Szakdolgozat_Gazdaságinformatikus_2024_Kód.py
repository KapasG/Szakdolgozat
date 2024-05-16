import pandas as pd
import numpy as np
from datetime import datetime as dt
import holidays
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
import time

from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
import copy
import tensorflow as tf
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
'''
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers.recurrent import LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor
import xlrd
from scipy.ndimage.filters import gaussian_filter


def Get_Data(load_loc,weather_loc):
    load=pd.read_excel(load_loc,header=0,usecols="A,J,K")
    load.columns=['Time', 'Load', 'MAVIR Est']
    load.Time=load.Time.apply(lambda x : dt.strptime(x[:13],'%Y.%m.%d %H'))+pd.Timedelta(hours=1)
    load=load.set_index('Time')
    est=load['MAVIR Est']
    load=pd.DataFrame(load.Load)

    weather=pd.read_csv(weather_loc,header=0)
    weather=weather.drop(labels=['metar','wxcodes', 'skyc1', 'skyc2', 'skyc3', 'skyc4','station'],axis=1)
    weather=weather.rename(columns={'valid':'Time'})
    weather.Time=weather.Time.apply(lambda x : dt.strptime(x[:13],'%Y-%m-%d %H'))
    weather=weather.set_index('Time')

    return load.join(weather, how='inner'),est

def To_Metric(df):
    df[['tmpf','dwpf','feel']]=df[['tmpf','dwpf','feel']].apply(lambda x: (x-32)*(5/9))
    df[['sknt','gust']]=df[['sknt','gust']].apply(lambda x: x*1.852)
    df[['p01i','alti']]=df[['p01i','alti']].apply(lambda x:x*25.4)
    df=df.rename(columns={'tmpf':'tmpc','dwpf':'dwpc','sknt':'skph','p01i':'p01mm','alti':'altcm'})
    return df

def DayDegree(df):
    tempgroup=df[['Load','tmpc']]
    tempgroup=tempgroup.groupby(by='tmpc').mean()
    minloaddeg=tempgroup[tempgroup.Load==tempgroup.Load.min()].index[0]
    '''
tempgroup=df[['Load','tmpc']]
tempgroup=tempgroup.groupby(by='tmpc').mean()
minloaddeg=tempgroup[tempgroup.Load==tempgroup.Load.min()].index[0]

fig, plt=pyplot.subplots()
plt.plot(tempgroup.index,tempgroup.Load, color='g')
plt.axvline(x = minloaddeg, color='C1', ls=':')
plt.set(ylabel='Átlagos Rendszerterhelés (MW)',xlabel='Hőmérséklet (°C)')
pyplot.xlim(left=2,right=37)
pyplot.show()

    '''
    return abs(df.tmpc-minloaddeg)




def TimeSetup(df):
    df=df.iloc[~df.index.duplicated()]
    df['Month']=df.index.month
    df['DoW']=df.index.dayofweek
    df['Hour']=df.index.hour
    df['Holiday']=df.reset_index().Time.apply(lambda x: x in holidays.HU(years=[*range(df.index[0].year,df.index[-1].year+1)])).tolist()
    df['ModDoW']=df.DoW
    df.loc[df.Holiday,'ModDoW']=6
    
    for i in range(7):
        df['ModDoW'+str(i)]=0
        df.loc[df.ModDoW==i,'ModDoW'+str(i)]=1
    return df.asfreq('h').fillna(0)

def FitArma(df):
    #daily = df.asfreq('D').fillna(0)
    #data=daily.drop(labels=daily.columns.tolist()[8:-4]+['tmpc','p01mm','Hour','Holiday'], axis=1)
    data=df
    o=(24,0,24)
    arma=ARIMA(endog=data['Load'], exog=data.iloc[:,1:], order=o)
    arma = arma.fit()
    print(arma.summary())
    #arma1 = ARIMA(endog=data['Load'], exog=data.iloc[:,1:].drop(labels=['drct','altcm','skph'],axis=1), order=o).fit()
    #print(arma1.summary())
    return arma

def ManualARMAX(model,lags,data,start,steps):
    load=data.Load[:start]
    for curr in range(start,start+steps):
        exog=model.params[1:data.columns.shape[0]]*data.iloc[curr,1:]
        AR=list(map(lambda x, y: x * y, list(reversed(list(data.Load[curr-lags:curr]))), list(model.arparams)))
        MA=list(map(lambda x, y: x * y, list(reversed(model.resid[curr-lags:curr])), list(model.maparams)))
        est=(sum(exog)+sum(AR)+sum(MA))-model.params[0]/4
        load[load.index[-1]+ pd.Timedelta(hours=1)]=est
    return load[start:]

def Plots(arma):
    # line plot of residuals
    residuals = pd.DataFrame(arma.resid)
    residuals.plot()
    pyplot.show()
    # density plot of residuals
    residuals.plot(kind='kde')
    pyplot.show()
    # summary stats of residuals
    print(residuals.describe())

    plot_acf(residuals, lags=48)
    pyplot.show() 


    plot_acf(df['Load'], lags=168*1.5)
    pyplot.axvline(x = 168, color='C1', ls=':')
    pyplot.ylim(bottom=-0.1)
    pyplot.title(None)
    pyplot.xlabel('Késleltetés (óra)')
    pyplot.ylabel('Korreláció')    
    pyplot.show()

    plot_pacf(df['Load'], lags=168*1.5)
    pyplot.show()

    sns.pairplot(df[df.Load>0].iloc[:,:4])
    pyplot.show()

def LSTMcreateXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i,1:])
        dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)

def build_LSTM(optimizer='adam'):
    grid_model = tf.keras.models.Sequential()
    grid_model.add(tf.keras.layers.LSTM(50,return_sequences=False,input_shape=(168,13)))
    grid_model.add(tf.keras.layers.Dropout(0.2))
    grid_model.add(tf.keras.layers.Dense(1))
    grid_model.compile(loss = 'mse',optimizer = optimizer)
    return grid_model

def build_GRU(optimizer='adam'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.GRU(16, input_shape=(168,13)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss = 'mse',optimizer = optimizer)
    return model


def FitLSTM(df):
    scaleddf = MinMaxScaler(feature_range=(0, 1)).fit_transform(df)
    train_size = int(len(scaleddf) * 0.8)
    traindf = scaleddf[:train_size]
    testdf = scaleddf[train_size:]

    lookback = 168
    trainX, trainY = LSTMcreateXY(traindf, lookback)
    testX, testY = LSTMcreateXY(testdf, lookback)

    grid_model = KerasRegressor(build_fn=build_LSTM, verbose=1)

    parameters = {'batch_size': [15,16,17], 'epochs': [11,12,13]}
    grid_search = GridSearchCV(estimator=grid_model,
                               param_grid=parameters,
                               cv=2,
                               verbose=1,
                               refit=True,
                               n_jobs=3)
    grid_search = grid_search.fit(trainX, trainY, validation_data=(testX, testY))  # Provide validation data
    return grid_search

def FitGRU(df):
    scaleddf = MinMaxScaler(feature_range=(0, 1)).fit_transform(df)
    train_size = int(len(scaleddf) * 0.8)
    traindf = scaleddf[:train_size]
    testdf = scaleddf[train_size:]

    lookback = 168
    trainX, trainY = LSTMcreateXY(traindf, lookback)
    testX, testY = LSTMcreateXY(testdf, lookback)

    grid_model = KerasRegressor(build_fn=build_GRU, verbose=1)

    parameters = {'batch_size': [15,16,17], 'epochs': [11,12,13]}
    grid_search = GridSearchCV(estimator=grid_model,
                               param_grid=parameters,
                               cv=2,
                               verbose=1,
                               refit=True,
                               n_jobs=3)
    grid_search = grid_search.fit(trainX, trainY, validation_data=(testX, testY))  # Provide validation data
    return grid_search

def PredLSTM(traindf,testdf,mod):
    transf=MinMaxScaler(feature_range=(0, 1))
    testdf = transf.fit_transform(testdf)
    traindf = transf.fit_transform(traindf)
    lookback = 168
    trainX, trainY = LSTMcreateXY(traindf, lookback)
    testX, testY = LSTMcreateXY(testdf, lookback)

    model=build_LSTM()
    model.fit(trainX, trainY,epochs=mod.param_epochs[0], batch_size=mod.param_batch_size[0], verbose=1)
    pred=model.predict(testX)

    testres=pd.DataFrame(testdf).iloc[168:,1:]
    testres.insert(loc=0,column='pred',value=pred)
    preddf=pd.DataFrame(transf.inverse_transform(testres))

    return model,preddf[0]

def PredGRU(traindf,testdf,mod):
    transf=MinMaxScaler(feature_range=(0, 1))
    testdf = transf.fit_transform(testdf)
    traindf = transf.fit_transform(traindf)
    lookback = 168
    trainX, trainY = LSTMcreateXY(traindf, lookback)
    testX, testY = LSTMcreateXY(testdf, lookback)

    model=build_GRU()
    model.fit(trainX, trainY,epochs=mod.param_epochs[0], batch_size=mod.param_batch_size[0], verbose=1)
    pred=model.predict(testX)

    testres=pd.DataFrame(testdf).iloc[168:,1:]
    testres.insert(loc=0,column='pred',value=pred)
    preddf=pd.DataFrame(transf.inverse_transform(testres))

    return model,preddf[0]

def Visualizations():
    daygroup=df[['Load','DoW']].groupby(by='DoW').mean()
    moddaygroup=df[['Load','ModDoW']].groupby(by='ModDoW').mean()

    fig, plt=pyplot.subplots()
    plt.bar(['Hétfő','Kedd','Szerda','Csütörtök','Péntek','Szombat','Vasárnap'],moddaygroup.Load)
    plt.set(ylabel='Átlagos Rendszerterhelés (MW)')
    pyplot.show()
    '''
    '''

    dategroup=df.Load.reset_index()
    dategroup.Time=dategroup.Time.apply(lambda x :x.strftime('%m-%d'))
    dategroup=dategroup.groupby(by="Time").mean()

    dategroup.reset_index().Load.plot().set(ylabel='Átlagos Rendszerterhelés (MW)',xlabel='Év Napja')

    fig, plt=pyplot.subplots()
    plt.plot(dategroup.index,dategroup.Load)
    plt.set(ylabel='Átlagos Rendszerterhelés (MW)',xlabel='Hőmérséklet (°C)')
    pyplot.show()



    dategroup=df.Load.reset_index()
    dategroup.Time=dategroup[(dategroup.Time<'2020')|(dategroup.Time>'2021')].Time.apply(lambda x :x.strftime('%m'))
    dategroup=dategroup.groupby(by="Time").mean()
    twentytwenty=df.Load.reset_index()
    twentytwenty.Time=twentytwenty[(twentytwenty.Time>'2020') & (twentytwenty.Time<'2021')].Time.apply(lambda x :x.strftime('%m'))
    twentytwenty=twentytwenty.groupby(by="Time").mean()
    fig, plt=pyplot.subplots()
    plt.plot(dategroup.index,dategroup.Load, color='red')
    plt.plot(twentytwenty.index,twentytwenty.Load)
    plt.set(ylabel='Abszolút Becslési Hiba (MW)')
    pyplot.show()

    autocorrelation_plot(df['Load'])
    pyplot.show()

    fig, plt=pyplot.subplots()
    plt.plot(lstmfor.index,abs(lstmfor-testdf[168:].Load))
    plt.set(ylabel='Abszolút Becslési Hiba (MW)')
    pyplot.show()

    dategroup=df.Load.reset_index()
    dategroup.Time=dategroup[(dategroup.Time<'2019.06')&(dategroup.Time>'2019.04')].Time.apply(lambda x :x.strftime('%H'))
    dategroup=dategroup.groupby(by="Time").mean()
    twentytwenty=df.Load.reset_index()
    twentytwenty.Time=twentytwenty[(twentytwenty.Time>'2020.04') & (twentytwenty.Time<'2020.06')].Time.apply(lambda x :x.strftime('%H'))
    twentytwenty=twentytwenty.groupby(by="Time").mean()
    fig, plt=pyplot.subplots()
    plt.plot(dategroup.index,dategroup.Load, label='2019 április-május', color='g')
    plt.plot(twentytwenty.index,twentytwenty.Load, label='2020 április-május', color='C1')
    plt.set(ylabel='Átlagos rendszerterhelés (MW)', xlabel='Nap órája')
    plt.legend()
    pyplot.show()

    fig, plt=pyplot.subplots()
    armafor.index=testdf.index
    ser=pd.Series()
    for x in range(0,len(armafor),168):
        ser[armafor.index[x]]=abs(armafor[x:x+168]-testdf.Load[x:x+168]).mean()
    ser.plot()
    plt.set(ylabel='Abszolút Becslési Hiba (MW)')
    pyplot.show()




#Import and format data
LoadLoc5year=r"Load_2019-2023.xlsx"
WeatherLoc5year=r"Idojaras_2019-2023.csv"
df=pd.DataFrame()
df,MAVIREst=Get_Data(LoadLoc5year,WeatherLoc5year)
df=To_Metric(df)
df=TimeSetup(df)
df['DayDegreeC']=DayDegree(df)

#Select variables and create train-test split
keepdf=df.drop(labels=df.columns.tolist()[8:-8]+['tmpc','p01mm','Hour','Holiday','DoW'], axis=1)
keepdf=keepdf[keepdf.Load>0]
train_size = int(len(keepdf) * 0.8)
#train_size=int(len(keepdf))-24*14
traindf = keepdf[:train_size]
testdf = keepdf[train_size:]


#Create ARMAX model and 1-year-ahead prediction
armatime=time.time()
arma=FitArma(traindf)
armafor=arma.predict(start=len(traindf), end=len(keepdf)-1,exog=testdf.iloc[:,1:])
armaRsq=r2_score(testdf.Load,armafor)
armatime=time.time()-armatime


#Create LSTM model and 1-year-ahead prediciton
search=FitLSTM(traindf)
res=pd.DataFrame(data=search.cv_results_)
opt=res[res.rank_test_score==1].reset_index(drop=True)
lstmtime=time.time()
lstm,lstmfor=PredLSTM(traindf=traindf,testdf=testdf,mod=opt)
lstmtime=time.time()-lstmtime
lstmfor.index=testdf.Load[168:].index
lstmRsq=r2_score(testdf.Load[168:],lstmfor)

#Create GRU model and 1-year-ahead prediciton
searchGRU=FitGRU(traindf)
resGRU=pd.DataFrame(data=searchGRU.cv_results_)
optGRU=res[resGRU.rank_test_score==1].reset_index(drop=True)
grutime=time.time()
GRU,GRUfor=PredGRU(traindf=traindf,testdf=testdf,mod=optGRU) 
GRUfor.index=testdf.Load[168:].index
gruRsq=r2_score(testdf.Load[168:],GRUfor)
grutime=time.time()-grutime

#Create hourly predictions for GRNNs
lstmpredtime=time.time()
transf=MinMaxScaler(feature_range=(0, 1))
scaledf = transf.fit_transform(keepdf)
x,y=LSTMcreateXY(scaledf,168)
lstmpreds=pd.Series()
for r in range(int(x.shape[0]*0.8),x.shape[0]):
    lstmpreds[keepdf.index[r+168]]=lstm.predict(x[r-1:r])[0][0]
preddf=pd.DataFrame(scaledf[int((scaledf.shape[0]-168)*0.8)+168:,1:])
preddf.insert(loc=0,column='pred',value=lstmpreds.values)
preddf=pd.DataFrame(transf.inverse_transform(preddf),index=lstmpreds.index)
lstmpreds=preddf[0]
lstmpredtime=time.time()-lstmpredtime

grupredtime=time.time()
transf=MinMaxScaler(feature_range=(0, 1))
scaledf = transf.fit_transform(keepdf)
x,y=LSTMcreateXY(scaledf,168)
grupreds=pd.Series()
for r in range(int(x.shape[0]*0.8),x.shape[0]):
    grupreds[keepdf.index[r+168]]=GRU.predict(x[r-1:r])[0][0]
preddf=pd.DataFrame(scaledf[int((scaledf.shape[0]-168)*0.8)+168:,1:])
preddf.insert(loc=0,column='pred',value=grupreds.values)
preddf=pd.DataFrame(transf.inverse_transform(preddf),index=grupreds.index)
grupreds=preddf[0]
grupredtime=time.time()-grupredtime



#Create 1-day-ahead predicitons with GRNNs
transf=MinMaxScaler(feature_range=(0, 1))
scaledf = transf.fit_transform(keepdf)
x,y=LSTMcreateXY(scaledf,168)
dpredlstm=pd.Series()
for r in range(int(x.shape[0]*0.8),x.shape[0],24):
    for i,pred in enumerate(list(map(lambda x: x[0], lstm.predict(x[r:r+24])))):
        dpredlstm[keepdf.index[r+i+168]]=pred
preddf=pd.DataFrame(scaledf[int((scaledf.shape[0]-168)*0.8)+168:,1:])
preddf.insert(loc=0,column='pred',value=dpredlstm.values)
preddf=pd.DataFrame(transf.inverse_transform(preddf),index=dpredlstm.index)
dpredlstm=preddf[0]

transf=MinMaxScaler(feature_range=(0, 1))
scaledf = transf.fit_transform(keepdf)
x,y=LSTMcreateXY(scaledf,168)
dpredgru=pd.Series()
for r in range(int(x.shape[0]*0.8),x.shape[0],24):
    for i,pred in enumerate(list(map(lambda x: x[0], GRU.predict(x[r:r+24])))):
        dpredgru[keepdf.index[r+i+168]]=pred
preddf=pd.DataFrame(scaledf[int((scaledf.shape[0]-168)*0.8)+168:,1:])
preddf.insert(loc=0,column='pred',value=dpredgru.values)
preddf=pd.DataFrame(transf.inverse_transform(preddf),index=dpredgru.index)
dpredgru=preddf[0]


#Create hourly and 1-day-ahead predictions with ARMAX
armapredtime=time.time()
armapreds=pd.Series()
for r in range(int(keepdf.shape[0]*0.8),keepdf.shape[0],1):
    armapreds=pd.concat([armapreds,ManualARMAX(arma,24,keepdf,r-1,1)])
armapredtime=time.time()-armapredtime

armapredtime=time.time()
armad=pd.Series()
for r in range(int(keepdf.shape[0]*0.8),keepdf.shape[0],24):
    armad=pd.concat([armad,ManualARMAX(arma,24,keepdf,r-1,24)])
armapredtime=time.time()-armapredtime

#Calculate errros
lstmerror=lstmpreds-testdf.Load[168:]
gruerror=grupreds-testdf.Load[168:]
errordelta=abs(gruerror)-abs(lstmerror)
lstmpcterror=lstmerror/testdf.Load[168:]
grupcterror=gruerror/testdf.Load[168:]
pctdelta=abs(lstmpcterror)-abs(grupcterror)

#Check performance on holidays
holidaylist=list(map(lambda x: x in holidays.HU(years=[*range(pctdelta.index.year.tolist()[0],pctdelta.index.year.tolist()[-1])]),pctdelta.index.tolist()))
pctdelta[holidaylist].mean()

slope, intercept = np.polyfit(range(0,errordelta.shape[0]),errordelta, 1)
pyplot.plot(testdf.Load[168:].index.tolist(),errordelta,label="Difference")
pyplot.plot(testdf.Load[168:].index.tolist(), slope*range(0,errordelta.shape[0]) + intercept, '-', label='Trend Line')
pyplot.axvspan(testdf.Load[168:].index.tolist()[1400], testdf.Load[168:].index.tolist()[1483], color='gray', alpha=0.3)
pyplot.show()


#Accuracy of MAVIR estimation
mavir=pd.DataFrame(MAVIREst)
kopy=keepdf.copy()
kopy=kopy.join(mavir,how='left')
kopy=kopy[kopy['MAVIR Est']>0]
r2_score(kopy.Load.iloc[int(keepdf.shape[0]*0.8):],kopy['MAVIR Est'].iloc[int(keepdf.shape[0]*0.8):])


#Compare accuracy metrics
mean_squared_error(keepdf.iloc[int((scaledf.shape[0]-168)*0.8)+168:,0],grupreds)
mean_squared_error(keepdf.iloc[int((scaledf.shape[0]-168)*0.8)+168:,0],dpredlstm)
mean_squared_error(keepdf.iloc[int((scaledf.shape[0]-168)*0.8)+168:,0],lstmpreds)
mean_squared_error(keepdf.iloc[int((scaledf.shape[0]-168)*0.8)+168:,0],dpredgru)
mean_squared_error(keepdf.iloc[int((scaledf.shape[0])*0.8)+168:,0],lstmfor)
mean_squared_error(keepdf.iloc[int((scaledf.shape[0])*0.8)+168:,0],GRUfor)
mean_squared_error(keepdf.Load.iloc[int(keepdf.shape[0]*0.8):],armafor)
mean_squared_error(keepdf.Load.iloc[int(keepdf.shape[0]*0.8):],armapreds)
mean_squared_error(keepdf.Load.iloc[int(keepdf.shape[0]*0.8):-19],armad)
mean_squared_error(kopy.Load.iloc[int(keepdf.shape[0]*0.8):],kopy['MAVIR Est'].iloc[int(keepdf.shape[0]*0.8):])

r2_score(keepdf.iloc[int((scaledf.shape[0]-168)*0.8)+168:,0],lstmpreds)
r2_score(keepdf.iloc[int((scaledf.shape[0]-168)*0.8)+168:,0],grupreds)
r2_score(keepdf.iloc[int((scaledf.shape[0]-168)*0.8)+168:,0],dpredlstm)
r2_score(keepdf.iloc[int((scaledf.shape[0]-168)*0.8)+168:,0],dpredgru)
r2_score(keepdf.iloc[int((scaledf.shape[0])*0.8)+168:,0],lstmfor)
r2_score(keepdf.iloc[int((scaledf.shape[0])*0.8)+168:,0],GRUfor)
r2_score(keepdf.Load.iloc[int(keepdf.shape[0]*0.8):],armafor)
r2_score(keepdf.Load.iloc[int(keepdf.shape[0]*0.8):],armapreds)
r2_score(keepdf.Load.iloc[int(keepdf.shape[0]*0.8):-19],armad)
r2_score(kopy.Load.iloc[int(keepdf.shape[0]*0.8):],kopy['MAVIR Est'].iloc[int(keepdf.shape[0]*0.8):])

#Create errors dataframe
forcasts=pd.DataFrame()
forcasts['LSTM']=dpredlstm
forcasts['GRU']=dpredgru
forcasts['ARMAX']=armad
forcasts=forcasts[:-20]

errors=pd.DataFrame()
for i in forcasts.columns.tolist():
    errors[i]=abs(forcasts.loc[:,i]-testdf.Load[-forcasts.shape[0]-20:-20])

def ResultVisualizations():
    errors['DoW']=errors.index.dayofweek
    dayavgerror=errors.groupby(by='DoW').mean()
    dayavgerror['Day']=['Hétfő','Kedd','Szerda','Csütörtök','Péntek','Szombat','Vasárnap']
    dayavgerror.set_index('Day').plot().set(ylabel='Abszolút Becslési Hiba (MW)',xlabel=None)
    pyplot.show()



    errors['Temp']=df.tmpc[-forcasts.shape[0]-21:-20]
    fig, axs = pyplot.subplots(1, 3, figsize=(15, 5))
    s=3

    heatmap, xedges, yedges = np.histogram2d(errors[errors.LSTM<600].Temp, errors[errors.LSTM<600].LSTM, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    heatmap = gaussian_filter(heatmap, sigma=s)
    axs[0].imshow(heatmap.T, extent=extent, origin='lower', aspect=0.08, cmap='plasma')
    axs[0].set_title('LSTM')


    heatmap, xedges, yedges = np.histogram2d(errors[errors.GRU<600].Temp, errors[errors.GRU<600].GRU, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    heatmap = gaussian_filter(heatmap, sigma=s)
    axs[1].imshow(heatmap.T, extent=extent, origin='lower', aspect=0.08, cmap='plasma')
    axs[1].set_title('GRU')
    axs[1].set_xlabel('Hőmérséklet (°C)')

    heatmap, xedges, yedges = np.histogram2d(errors[errors.ARMAX<600].Temp, errors[errors.ARMAX<600].ARMAX, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    heatmap = gaussian_filter(heatmap, sigma=s)
    axs[2].imshow(heatmap.T, extent=extent, origin='lower', aspect=0.08, cmap='plasma')
    axs[2].set_title('ARMAX')
    pyplot.show()

    avgerrors=pd.DataFrame()
    for i in range(0,errors.shape[0],24*30):
        avgerrors[errors.index[-i]]=errors[-i-24*30:-i].mean()
    avgerrors=avgerrors.T
    avgerrors.plot()
    pyplot.show()
    errors.plot()
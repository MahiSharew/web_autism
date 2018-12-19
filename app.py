import numpy as np
import keras.models
import sys 
import os

from flask import Flask ,request, render_template,url_for
import librosa , librosa.display
sys.path.append(os.path.abspath("./model"))
from load import * 

app =Flask(__name__)

global model, graph 
model, graph =init()



@app.route("/")
@app.route("/index")
def index():
	return render_template('home.html')


@app.route("/predict",methods=["GET","POST"])
def predict():
	posts =[]
	response = {}
	test_dir = os.path.join('testData', 'td')
	testdata = testData(test_dir, r)
	#testdata=np.load(featurestest)
	for data in testdata:
	    x=np.reshape(data,(1,193,1))
	    with graph.as_default():
		#perform the prediction
			predication = model.predict(x)
	    #predication=model.predict(x)
    	#response['predictions'].append(prediction)
	    #print(predication)
	    #posts.append(predication);
	    	jk=(np.amax(predication)*100)
	    #print(jk)
	    #print(decode(predication) +' : '+ str(jk))
	return render_template('home.html',posts=predication)


import tqdm
def testData(file_path,file_ext='*.wav'):
    featurestest = np.empty((0,193))
   
    for fn in os.listdir(os.path.join('testData', 'td')):
        path = os.path.join(file_path,fn)
        print(path)
        try:
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(path)
        except Exception as e:
            print("[Error] extract feature error. %s" % (e))
            continue
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        featurestest = np.vstack([featurestest,ext_features])
        # labels = np.append(labels, fn.split('/')[1])
        #labelstest = np.append(labelstest, fn)
    print("extract %s features done")
    return np.array(featurestest)

def extract_feature(file_name):
    
    #X, sample_rate = sf.read(file_name, dtype='float32')
    X, sample_rate = librosa.load(file_name)
    if X.ndim > 1:
        X = X[:,0]
        print(X)
    X = X.T

    # short term fourier transform
    stft = np.abs(librosa.stft(X))

    # mfcc
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz




if __name__=='__main__':
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	#app.run(debug=True)



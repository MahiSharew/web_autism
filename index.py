from flask import Flask ,request, render_template,url_for
app =Flask(__name__)
from formpy.forms import RegistrationForm,LoginForm
app.config['SECRET_KEY']='ABCDFGERTDFGDFGDFGDFG'
posts=[{'author':'mahlet','title':'something','content':'blog _one '},
{'author':'mahlet1','title':'something1','content':'blog _one '}]
import sys
import os

from load import * 
from flask import Flask ,request, render_template,url_for
import librosa , librosa.display
sys.path.append(os.path.abspath("./model"))




global model ,graph
model,graph =init()


@app.route("/")
@app.route("/home")
def home():
	return render_template('home.html',posts=posts)


@app.route("/about")
def about():
	return render_template('about.html')

@app.route("/register")
def register():
	form=RegistrationForm()
	return render_template('register.html', form=form)

@app.route("/login")
def login():
	form=LoginForm()
	return render_template('login.html', form=form)







@app.route("/predication")
def predict():
	posts =[]
	test_dir = os.path.join('testData', 'td')
	testdata = testData(test_dir)
	#testdata=np.load(featurestest)
	for data in testdata:
		x=np.reshape(data,(1,193,1))
		with graph.as_default():
			predication=model.predict(x)
			print(predication)
			#posts.append(predication)
			accuracy=(np.amax(predication)*100)
			#print(jk)
			value=decode(predication) +' : '+ str(accuracy)
			print(value)
			#response = np.array_str(np.argmax(predication,axis=1))
			posts.append(predication)
			#return response	
			
	return render_template('index.html',posts=posts)

def decode(datum):
    if(np.argmax(datum)==0):  return 'ASD'
    elif(np.argmax(datum)==1): return 'baby Laugh'
    elif(np.argmax(datum)==2): return 'baby Cry'
    elif(np.argmax(datum)==3): return 'noise'
    elif(np.argmax(datum)==4): return 'silence'
    elif(np.argmax(datum)==5): return 'TD'
    else: return 'Normal'

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
	app.run(debug=True)
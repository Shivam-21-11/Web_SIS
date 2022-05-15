from flask import Flask , render_template,request,flash,redirect,url_for
import tensorflow as tf 
from werkzeug.utils import secure_filename
import os
import numpy as np
app = Flask(__name__)

upload_folder = 'static/uploads'
pred_folder = 'static/preds'
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['PRED_FOLDER'] = pred_folder
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model = tf.keras.models.load_model('custom_unet_v1_13-04-22.h5')
##################helper functions ###############################
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(path):
    '''
    Preprocess the image so it is compatible with the model.
    Our model input takes shape (1, 128, 128, 3)
    '''
    img = tf.keras.utils.load_img(path) 
    img = tf.keras.preprocessing.image.smart_resize(img,(128,128))
    img = tf.keras.utils.img_to_array(img)
    img = img/255.0
    img = np.array([img])
    return img

def create_mask(pred_mask):
    '''
    Create a mask from the prediction.
    '''
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]
######################################################
@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')  

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = preprocess_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            pred_mask = model.predict(img)
            pred_mask = create_mask(pred_mask)
            tf.keras.preprocessing.image.save_img(os.path.join(app.config['PRED_FOLDER'], filename),pred_mask)
            return render_template('pred.html',filename=filename)
        else:
            flash('File extension not allowed')
            return redirect(request.url)


@app.route('/display/<filename>')
def display(filename):
    return redirect(url_for('static',filename='uploads/'+filename),code=301)

@app.route('/displaypred/<filename>')
def displaypred(filename):
    return redirect(url_for('static',filename='preds/'+filename),code=301)

if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True,port=3000)
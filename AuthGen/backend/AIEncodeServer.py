import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import bchlib
from PIL import Image,ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from io import BytesIO

app = Flask(__name__)
CORS(app)

# handle error
@app.errorhandler(400)
def not_found_error(error):
    return jsonify({'error': 'Bad request'}), 400


@app.route('/make_image', methods=['POST'])
async def make_image():

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    secret = request.form['secret']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:

        image = Image.open(BytesIO(file.read()))

        BCH_POLYNOMIAL = 137 #137
        BCH_BITS = 5 #5

        sess = tf.compat.v1.InteractiveSession(graph=tf.Graph())

        model = tf.compat.v1.saved_model.load(sess, [tag_constants.SERVING], 'saved_models/stegastamp_pretrained')

        input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
        input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
        input_secret = tf.compat.v1.get_default_graph().get_tensor_by_name(input_secret_name)
        input_image = tf.compat.v1.get_default_graph().get_tensor_by_name(input_image_name)

        output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
        output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
        output_stegastamp = tf.compat.v1.get_default_graph().get_tensor_by_name(output_stegastamp_name)
        output_residual = tf.compat.v1.get_default_graph().get_tensor_by_name(output_residual_name)

        width = 400
        height = 400

        bch = bchlib.BCH(prim_poly = BCH_POLYNOMIAL, t = BCH_BITS)
        data = bytearray(secret + ' '*(7-len(secret)), 'utf-8')
        ecc = bch.encode(data)
        packet = data+ecc
        packet_binary = ''.join(format(x, '08b') for x in packet)
        secret = [int(x) for x in packet_binary]
        secret.extend([0,0,0,0])

        size = (width, height)


        image = image.convert("RGB")
        image = np.array(ImageOps.fit(image,size),dtype=np.float32)
        image /= 255.

        feed_dict = {input_secret:[secret],
                         input_image:[image]}

        hidden_img, residual = sess.run([output_stegastamp, output_residual],feed_dict=feed_dict)

        rescaled = (hidden_img[0] * 255).astype(np.uint8)
        raw_img = (image * 255).astype(np.uint8)

        im = Image.fromarray(np.array(rescaled))

        file_path = "out/out.jpg"
        im.save(file_path)

        return send_file(file_path, as_attachment=True)    

if __name__ == '__main__':
    app.run(debug=True, port=7500)

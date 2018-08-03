from tensorflow_cnn_train import crack_captcha_cnn
from tensorflow_cnn_train import convert2gray
from tensorflow_cnn_train import vec2text
from tensorflow_cnn_train import gen_captcha_text_and_image

from tensorflow_cnn_train import MAX_CAPTCHA
from tensorflow_cnn_train import CHAR_SET_LEN
from tensorflow_cnn_train import X
from tensorflow_cnn_train import keep_prob

import time
import tensorflow as tf
import numpy as np

def crack_captcha(captcha_image):

    predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)

    text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1}) # 对矩阵按行或列计算最大值

    text = text_list[0].tolist()        # 得到的是vec
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    i = 0
    for n in text:
        vector[i * CHAR_SET_LEN + n] = 1
        i += 1
    return vec2text(vector)

if __name__ == '__main__':
    output = crack_captcha_cnn()
    saver = tf.train.Saver()
    sess = tf.Session() 
    saver.restore(sess, "./model/crack_capcha.model-6900") 
    '''
    注意，这里自行加上训练出来的模型的步数，比如我训练了6900步，那我自己加上***.model-6900
    简单来说就是要对应model文件夹里边的model文件名
    '''
    while(1):
        i, j = 0, 0
        start = time.time()
        text, image = gen_captcha_text_and_image('test')

        image = convert2gray(image).flatten() / 255     #一维化
        start = time.time()
        predict_text = crack_captcha(image)
        end = time.time()
        print("正确: {}  预测: {}  时间:{}".format(text, predict_text, end-start))
        for txt in text:
            if txt in predict_text[j]:
                i += 1
            else:
                pass
            j += 1
        print("识别率：" + str(i/4))

# HZNU_Captcha_CNN
HZNU的教务处系统验证码识别

## How to use
1.Unzip `HZNU_Captcha_Images.zip`, it contain 2370 train images and 624 images for test.   
2.Bash:   
```
python tensorflow_cnn_train.py
```
3.Wait   
4.Test:   
```
python test_model.py
```
## The Result
![Result](https://github.com/HytonightYX/HZNU_Captcha_CNN/blob/master/Tensorboard_res.png)    
 So we got nearly 100% accuracy. It's really a good result!

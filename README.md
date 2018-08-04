# HZNU_Captcha_CNN
HZNU的教务处系统验证码识别，理论上可以识别所有方正教务系统的验证码

## How to use
#### 1.Unzip `HZNU_Captcha_Images.zip`, it contain 2370 train images and 624 images for test.   
#### 2.Bash:   
```
python tensorflow_cnn_train.py
```
#### 3.Wait
I used `i5-7200U 8GB DDR4 MX-150`, and trained this model for anout 20min.
#### 4.Test:   
```
python test_model.py
```
## The Train Result
![ So we got nearly 100% accuracy](https://github.com/HytonightYX/HZNU_Captcha_CNN/blob/master/Tensorboard_res.png)    
**Our model is saved in ** `model.zip` **you can also train it again.**
## The Test Result(A small part of console)
```
···
正确: 6tvs  预测: 6tvs[识别正确]
正确: fj2c  预测: fj2c[识别正确]
正确: 688k  预测: 688k[识别正确]
正确: 44f5  预测: k4f5[识别错误]
正确: kl54  预测: kl54[识别正确]
正确: x16a  预测: x16a[识别正确]
正确: upj4  预测: upj4[识别正确]
正确: iu8r  预测: iu8r[识别正确]
正确: wcuj  预测: wcuj[识别正确]
正确: midw  预测: midw[识别正确]
正确: aw0a  预测: aw0a[识别正确]
正确: v015  预测: v015[识别正确]
正确: n631  预测: n631[识别正确]
正确: ukbl  预测: ukbl[识别正确]
正确: 4fuk  预测: 4fuk[识别正确]
正确: u41t  预测: u41t[识别正确]
正确: yugp  预测: yugp[识别正确]
正确: b0uc  预测: b0uc[识别正确]
```

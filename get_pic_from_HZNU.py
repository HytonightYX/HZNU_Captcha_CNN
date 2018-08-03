import requests
import os
import re

for i in range(3000): # 设定爬取得验证码总数
    response = requests.get('http://jwgl1.hznu.edu.cn/CheckCode.aspx') # 网址，理论上所有的方正系统都行，修改hznu为所在大学即可
    with open(str(i+1) +'.png', 'wb') as f:
        f.write(response.content)
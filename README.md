# PTIT
PTIT- Phát hiện lỗ hổng trong source-code PHP
# Giới thiệu 
Mô hình phát hiện lỗ hổng trong source-code PHP 
# Dataset 
103	PHP Vulnerability Test Suite; 114	PHP test suite - XSS, SQLi 1.0.0:
https://samate.nist.gov/SARD/test-suites 
SQLi Labs: https://github.com/Audi-1/sqli-labs
XSS lab:  https://github.com/PwnAwan/XSS-Labs
OWASP : https://github.com/KaitoRyouga/OWASP/tree/master
# Train
Chuyển đổi mã nguồn thành tokens:
php Tokenziez.php
# Các mô hình thử nghiệm:
python BiLSTM.py
python GRU.py
python XLLSTM.py
python Transformer.py
python DataBalancing.py


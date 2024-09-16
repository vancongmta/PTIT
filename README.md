# PTIT
PTIT- Phát hiện lỗ hổng trong source-code PHP
## Giới thiệu 
Mô hình phát hiện lỗ hổng trong source-code PHP 
## Dataset 
103	PHP Vulnerability Test Suite; 114	PHP test suite - XSS, SQLi 1.0.0:

https://samate.nist.gov/SARD/test-suites

SQLi Labs: https://github.com/Audi-1/sqli-labs

XSS lab:  https://github.com/PwnAwan/XSS-Labs

OWASP : https://github.com/KaitoRyouga/OWASP/tree/master


## Train
Chuyển đổi mã nguồn thành tokens:
```
php Tokenziez.php
```
## Các mô hình thử nghiệm:
BiLSTM (Bidirectional Long Short-Term Memory) là một biến thể của mô hình LSTM, trong đó mô hình sẽ được huấn luyện theo cả hai hướng,
```
python BiLSTM.py
```
GRU (Gated Recurrent Unit) là một loại mạng nơ-ron hồi quy tương tự như LSTM nhưng có kiến trúc đơn giản hơn. Nó chỉ có hai cổng chính: cổng cập nhật và cổng quên. GRU giúp khắc phục các vấn đề về biến mất hoặc bùng nổ gradient như LSTM, nhưng với cấu trúc gọn nhẹ hơn
```
python GRU.py
```
XLLSTM là một biến thể mở rộng của LSTM (Extended LSTM) với các cải tiến về cấu trúc và kỹ thuật huấn luyện. Mục tiêu của XLLSTM là khắc phục các hạn chế của LSTM thông thường bằng cách tối ưu hóa việc huấn luyện trên các chuỗi dài và phức tạp hơn
```
python XLLSTM.py
```
Transformer là một mô hình hoàn toàn khác biệt so với các mô hình RNN (LSTM, GRU). Nó được giới thiệu trong nghiên cứu nổi tiếng "Attention is All You Need" và đã trở thành mô hình mạnh mẽ nhất trong nhiều lĩnh vực xử lý ngôn ngữ tự nhiên và thị giác máy tính. Thay vì sử dụng mạng nơ-ron hồi quy, Transformer dựa trên cơ chế attention để mô hình hóa các mối quan hệ giữa các phần tử trong chuỗi đầu vào, bất kể khoảng cách giữa chúng.
```
python Transformer.py
```
thuật toán cân bằng dữ liệu (Data Balancing). Trong các bài toán học máy, đặc biệt là các bài toán phân loại, dữ liệu thường không đồng đều, tức là có sự chênh lệch lớn giữa số lượng mẫu của các lớp. Điều này dẫn đến mô hình có xu hướng thiên lệch về lớp có số lượng mẫu nhiều hơn. Data Balancing nhằm mục đích cân bằng tỷ lệ giữa các lớp bằng các kỹ thuật như undersampling, oversampling, hoặc sử dụng các thuật toán nâng cao như SMOTE (Synthetic Minority Over-sampling Technique).
```
python DataBalancing.py
```

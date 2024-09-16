# PTIT
PTIT- Phát hiện lỗ hổng trong source-code PHP
## Giới thiệu 
PTIT là một mô hình được phát triển để phát hiện lỗ hổng trong mã nguồn PHP, với mục tiêu nâng cao hiệu quả phát hiện các vấn đề bảo mật trong ứng dụng web. Mô hình sử dụng nhiều phương pháp hiện đại để phân tích và phát hiện các loại lỗ hổng phổ biến trong mã nguồn PHP như SQL Injection (SQLi) và Cross-Site Scripting (XSS).
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
Điều chỉnh tokens
| Duplicate Tokens                                        | Token Combinations                         |
|---------------------------------------------------------|--------------------------------------------|
| T_UNDEFINED_VAR , T_UNDEFINED_VAR , T_UNDEFINED_VAR , T_UNDEFINED_VAR | 4_T_UNDEFINED_VAR                         |
| T_ECHO T_CONSTANT_ENCAPSED_STRING ; T_ECHO T_CONSTANT_ENCAPSED_STRING ; | [ 3_T_ECHO T_CONSTANT_ENCAPSED_STRING ; ] |
| T_ECHO T_CONSTANT_ENCAPSED_STRING ; T_ECHO T_CONSTANT_ENCAPSED_STRING ; T_ECHO T_CONSTANT_ENCAPSED_STRING ; | [ 3_T_ECHO T_CONSTANT_ENCAPSED_STRING ; ] |
| T_CONSTANT_ENCAPSED_STRING , T_CONSTANT_ENCAPSED_STRING | 2_T_CONSTANT_ENCAPSED_STRING              |
| [ T_CONSTANT_ENCAPSED_STRING  . ” T_CONSTANT_ENCAPSED_STRING ]  | [ 2_T_CONSTANT_ENCAPSED_STRING .’’ ]      |
| T_CONSTANT_ENCAPSED_STRING , T_CONSTANT_ENCAPSED_STRING , T_CONSTANT_ENCAPSED_STRING , | 3_T_CONSTANT_ENCAPSED_STRING             |
| T_IGNORE , T_IGNORE , T_IGNORE ,                       | 3_T_IGNORE                                |
| T_ASSIGNMENT , T_ASSIGNMENT , T_ASSIGNMENT ,            | 3_T_ASSIGNMENT                            |
| T_COMPARISON , T_COMPARISON , T_COMPARISON ,            | 3_T_COMPARISON                            |
| T_INCLUDES , T_INCLUDES                                | 2_T_INCLUDES                              |
| T_ECHO . ’’ T_ECHO . ’’ T_ECHO . ’’                    | 3_T_ECHO . ’’                             |
| T_INPUT  ;  T_INPUT  ; T_INPUT ;                        | 3_T_INPUT                                 |



```
python Conversion.py
```
Run PTIT
```
python trainLSTM.py
```
Đánh giá mô hình

```
python confusion_matrix.py
```
## Các mô hình khác thử nghiệm:
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
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/866e34f3-9e1c-461f-9531-3c66b5755ce2" width="300"/>
      <p>Ma trận chuẩn hóa của TAP</p>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/b98f075e-29fa-4682-ba8f-24d2743db476" width="300"/>
      <p>Ma trận chuẩn hóa của PTIT</p>
    </td>
  </tr>
</table>





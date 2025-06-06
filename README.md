Training baseline model...
Training CUR model...

#####  Outputs   #######

--- Comparison of Baseline vs CUR ---
Model                    RMSE      MAE       
Baseline (No CUR)        86.69     67.28
CUR Decomposition        62.59     46.98
Frobenius Norm Relative Error: 0.2285
![image](https://github.com/user-attachments/assets/ebfb1f8a-3675-4e66-803a-288317910acd)



Frobenius Norm Relative Error: 0.9838
RMSE: 45.72
MAE: 35.88
![image](https://github.com/user-attachments/assets/b8caa1dd-5ffe-4a0d-aa46-987f120e1099)



Frobenius Norm Relative Error: 0.7094

--- Model Evaluation ---
RMSE: 39.70
MAE: 22.96
R² Score: 0.9124
![image](https://github.com/user-attachments/assets/bb381efc-8ab5-4e2c-9673-3b5cb0dce37a)



Frobenius Norm Relative Error: 0.2865

--- Final Model Evaluation ---
RMSE: 25.82
MAE: 17.15
R² Score: 0.9262
![image](https://github.com/user-attachments/assets/ed8dc700-fc5b-4a22-8a4e-c3e11ef8990a)



--- Model Evaluation ---
RMSE: 45.44
MAE: 28.14
R² Score: 0.8853
![image](https://github.com/user-attachments/assets/ca0c208e-bf7a-4a49-97ce-5f5fd26327e3)



--- Model Evaluation (SVR) ---
RMSE: 43.84
MAE: 23.36
R² Score: 0.8932
![image](https://github.com/user-attachments/assets/65f362ad-6c8d-4036-b17c-6d4a38c89a97)






📊 Model Performance Comparison (Best Values Highlighted)
Model	                RMSE	      MAE	    Time Required in sec.
0	RAW   	            86.21	      67.05	      10
1	RAW + CUR          	62.12	      46.23	      12
2	CUR 1	              46.56	      36.78      	0
3	CUR 2             	39.70     	22.96	      52
4	CUR 3             	25.82	      17.15     	21
5	PCA	                45.44	      28.14	      95
6	SVM	                43.84	      23.36	      755



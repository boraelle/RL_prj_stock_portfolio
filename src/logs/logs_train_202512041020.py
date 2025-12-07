======================================================================
🎯 RL Portfolio Management Training
   Algorithm: PPO + LSTM
   Assets: 14 ETFs + Cash
   Period: 2024년 1월 ~ 2025년 11월 (23개월)
======================================================================
✅ 데이터 로드 완료: 4,886개 레코드
   기간: 2024-01-02 ~ 2025-11-28
   거래일 수: 465일 (약 23.2개월)
   종목 수: 12개
✅ 환경 생성 완료
   Observation Space: Dict with 4 modals
   Action Space: (13,)
   Episode Length: 445일

🚀 PPO 모델 생성 중...
Using cpu device
✅ 모델 생성 완료
   총 파라미터: 264,619개

🎓 학습 시작 (총 150,000 timesteps)...
   데이터: 2024.01 ~ 2025.11 (23개월)
   예상 시간: 150분 (Mac CPU 기준)
   TensorBoard: tensorboard --logdir results/tensorboard

Logging to results/tensorboard/PPO_2
-----------------------------
| time/              |      |
|    fps             | 10   |
|    iterations      | 1    |
|    time_elapsed    | 187  |
|    total_timesteps | 2048 |
-----------------------------
---------------------------------------
| time/                   |           |
|    fps                  | 6         |
|    iterations           | 2         |
|    time_elapsed         | 647       |
|    total_timesteps      | 4096      |
| train/                  |           |
|    approx_kl            | 816.61646 |
|    clip_fraction        | 1         |
|    clip_range           | 0.2       |
|    entropy_loss         | -18.4     |
|    explained_variance   | 0.686     |
|    learning_rate        | 0.0003    |
|    loss                 | 3.9e+03   |
|    n_updates            | 10        |
|    policy_gradient_loss | 0.307     |
|    std                  | 1         |
|    value_loss           | 8.9e+04   |
---------------------------------------
--------------------------------------
| time/                   |          |
|    fps                  | 1        |
|    iterations           | 3        |
|    time_elapsed         | 5552     |
|    total_timesteps      | 6144     |
| train/                  |          |
|    approx_kl            | 483.2171 |
|    clip_fraction        | 1        |
|    clip_range           | 0.2      |
|    entropy_loss         | -18.5    |
|    explained_variance   | 0.895    |
|    learning_rate        | 0.0003   |
|    loss                 | 690      |
|    n_updates            | 20       |
|    policy_gradient_loss | 0.333    |
|    std                  | 1        |
|    value_loss           | 2.23e+03 |
--------------------------------------
---------------------------------------
| time/                   |           |
|    fps                  | 1         |
|    iterations           | 4         |
|    time_elapsed         | 5744      |
|    total_timesteps      | 8192      |
| train/                  |           |
|    approx_kl            | 361.94904 |
|    clip_fraction        | 1         |
|    clip_range           | 0.2       |
|    entropy_loss         | -18.5     |
|    explained_variance   | 0.878     |
|    learning_rate        | 0.0003    |
|    loss                 | 118       |
|    n_updates            | 30        |
|    policy_gradient_loss | 0.337     |
|    std                  | 1.01      |
|    value_loss           | 358       |
---------------------------------------
---------------------------------------
| time/                   |           |
|    fps                  | 1         |
|    iterations           | 5         |
|    time_elapsed         | 5937      |
|    total_timesteps      | 10240     |
| train/                  |           |
|    approx_kl            | 6182.0396 |
|    clip_fraction        | 1         |
|    clip_range           | 0.2       |
|    entropy_loss         | -18.6     |
|    explained_variance   | 0.879     |
|    learning_rate        | 0.0003    |
|    loss                 | 22        |
|    n_updates            | 40        |
|    policy_gradient_loss | 0.35      |
|    std                  | 1.02      |
|    value_loss           | 95.2      |
---------------------------------------
---------------------------------------
| time/                   |           |
|    fps                  | 2         |
|    iterations           | 6         |
|    time_elapsed         | 6127      |
|    total_timesteps      | 12288     |
| train/                  |           |
|    approx_kl            | 1311.4441 |
|    clip_fraction        | 1         |
|    clip_range           | 0.2       |
|    entropy_loss         | -18.9     |
|    explained_variance   | 0.874     |
|    learning_rate        | 0.0003    |
|    loss                 | 7.41      |
|    n_updates            | 50        |
|    policy_gradient_loss | 0.325     |
|    std                  | 1.06      |
|    value_loss           | 23.1      |
---------------------------------------
---------------------------------------
| time/                   |           |
|    fps                  | 2         |
|    iterations           | 7         |
|    time_elapsed         | 6319      |
|    total_timesteps      | 14336     |
| train/                  |           |
|    approx_kl            | 708.75385 |
|    clip_fraction        | 1         |
|    clip_range           | 0.2       |
|    entropy_loss         | -19.7     |
|    explained_variance   | 0.839     |
|    learning_rate        | 0.0003    |
|    loss                 | 2.06      |
|    n_updates            | 60        |
|    policy_gradient_loss | 0.324     |
|    std                  | 1.15      |
|    value_loss           | 4.73      |
---------------------------------------
--------------------------------------
| time/                   |          |
|    fps                  | 2        |
|    iterations           | 8        |
|    time_elapsed         | 6511     |
|    total_timesteps      | 16384    |
| train/                  |          |
|    approx_kl            | 592.5564 |
|    clip_fraction        | 1        |
|    clip_range           | 0.2      |
|    entropy_loss         | -21.1    |
|    explained_variance   | 0.822    |
|    learning_rate        | 0.0003   |
|    loss                 | 0.465    |
|    n_updates            | 70       |
|    policy_gradient_loss | 0.322    |
|    std                  | 1.3      |
|    value_loss           | 0.883    |
--------------------------------------
---------------------------------------
| time/                   |           |
|    fps                  | 2         |
|    iterations           | 9         |
|    time_elapsed         | 6704      |
|    total_timesteps      | 18432     |
| train/                  |           |
|    approx_kl            | 407.12506 |
|    clip_fraction        | 1         |
|    clip_range           | 0.2       |
|    entropy_loss         | -22.8     |
|    explained_variance   | 0.757     |
|    learning_rate        | 0.0003    |
|    loss                 | 0.149     |
|    n_updates            | 80        |
|    policy_gradient_loss | 0.29      |
|    std                  | 1.5       |
|    value_loss           | 0.269     |
---------------------------------------
---------------------------------------
| time/                   |           |
|    fps                  | 2         |
|    iterations           | 10        |
|    time_elapsed         | 6898      |
|    total_timesteps      | 20480     |
| train/                  |           |
|    approx_kl            | 238.37553 |
|    clip_fraction        | 0.997     |
|    clip_range           | 0.2       |
|    entropy_loss         | -24.6     |
|    explained_variance   | 0.702     |
|    learning_rate        | 0.0003    |
|    loss                 | 0.0649    |
|    n_updates            | 90        |
|    policy_gradient_loss | 0.305     |
|    std                  | 1.72      |
|    value_loss           | 0.0795    |
---------------------------------------
--------------------------------------
| time/                   |          |
|    fps                  | 3        |
|    iterations           | 11       |
|    time_elapsed         | 7093     |
|    total_timesteps      | 22528    |
| train/                  |          |
|    approx_kl            | 47.09996 |
|    clip_fraction        | 0.861    |
|    clip_range           | 0.2      |
|    entropy_loss         | -25.9    |
|    explained_variance   | 0.751    |
|    learning_rate        | 0.0003   |
|    loss                 | 0.116    |
|    n_updates            | 100      |
|    policy_gradient_loss | 0.337    |
|    std                  | 1.8      |
|    value_loss           | 0.0256   |
--------------------------------------
---------------------------------------
| time/                   |           |
|    fps                  | 3         |
|    iterations           | 12        |
|    time_elapsed         | 7286      |
|    total_timesteps      | 24576     |
| train/                  |           |
|    approx_kl            | 22.880236 |
|    clip_fraction        | 0.936     |
|    clip_range           | 0.2       |
|    entropy_loss         | -26.1     |
|    explained_variance   | 0.544     |
|    learning_rate        | 0.0003    |
|    loss                 | -0.164    |
|    n_updates            | 110       |
|    policy_gradient_loss | 0.203     |
|    std                  | 1.81      |
|    value_loss           | 0.00902   |
---------------------------------------
---------------------------------------
| time/                   |           |
|    fps                  | 3         |
|    iterations           | 13        |
|    time_elapsed         | 7480      |
|    total_timesteps      | 26624     |
| train/                  |           |
|    approx_kl            | 0.5070187 |
|    clip_fraction        | 0.111     |
|    clip_range           | 0.2       |
|    entropy_loss         | -26.2     |
|    explained_variance   | 0.0379    |
|    learning_rate        | 0.0003    |
|    loss                 | -0.211    |
|    n_updates            | 120       |
|    policy_gradient_loss | 0.0209    |
|    std                  | 1.82      |
|    value_loss           | 0.00385   |
---------------------------------------
------------------------------------------
| time/                   |              |
|    fps                  | 3            |
|    iterations           | 14           |
|    time_elapsed         | 7670         |
|    total_timesteps      | 28672        |
| train/                  |              |
|    approx_kl            | 0.0075376537 |
|    clip_fraction        | 0.0874       |
|    clip_range           | 0.2          |
|    entropy_loss         | -26.2        |
|    explained_variance   | 0.178        |
|    learning_rate        | 0.0003       |
|    loss                 | -0.27        |
|    n_updates            | 130          |
|    policy_gradient_loss | -0.00712     |
|    std                  | 1.82         |
|    value_loss           | 0.00267      |
------------------------------------------
----------------------------------------
| time/                   |            |
|    fps                  | 3          |
|    iterations           | 15         |
|    time_elapsed         | 7865       |
|    total_timesteps      | 30720      |
| train/                  |            |
|    approx_kl            | 0.00902248 |
|    clip_fraction        | 0.0861     |
|    clip_range           | 0.2        |
|    entropy_loss         | -26.2      |
|    explained_variance   | 0.321      |
|    learning_rate        | 0.0003     |
|    loss                 | -0.249     |
|    n_updates            | 140        |
|    policy_gradient_loss | -0.00736   |
|    std                  | 1.82       |
|    value_loss           | 0.00227    |
----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 4           |
|    iterations           | 16          |
|    time_elapsed         | 8059        |
|    total_timesteps      | 32768       |
| train/                  |             |
|    approx_kl            | 0.008790311 |
|    clip_fraction        | 0.0942      |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.3       |
|    explained_variance   | 0.452       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.244      |
|    n_updates            | 150         |
|    policy_gradient_loss | -0.00927    |
|    std                  | 1.83        |
|    value_loss           | 0.00197     |
-----------------------------------------
------------------------------------------
| time/                   |              |
|    fps                  | 4            |
|    iterations           | 17           |
|    time_elapsed         | 8251         |
|    total_timesteps      | 34816        |
| train/                  |              |
|    approx_kl            | 0.0113096535 |
|    clip_fraction        | 0.106        |
|    clip_range           | 0.2          |
|    entropy_loss         | -26.3        |
|    explained_variance   | 0.493        |
|    learning_rate        | 0.0003       |
|    loss                 | -0.256       |
|    n_updates            | 160          |
|    policy_gradient_loss | -0.00947     |
|    std                  | 1.82         |
|    value_loss           | 0.00165      |
------------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 4           |
|    iterations           | 18          |
|    time_elapsed         | 8446        |
|    total_timesteps      | 36864       |
| train/                  |             |
|    approx_kl            | 0.008680561 |
|    clip_fraction        | 0.0976      |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.2       |
|    explained_variance   | 0.458       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.28       |
|    n_updates            | 170         |
|    policy_gradient_loss | -0.00964    |
|    std                  | 1.82        |
|    value_loss           | 0.00125     |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 4           |
|    iterations           | 19          |
|    time_elapsed         | 8638        |
|    total_timesteps      | 38912       |
| train/                  |             |
|    approx_kl            | 0.008862877 |
|    clip_fraction        | 0.0787      |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.2       |
|    explained_variance   | 0.48        |
|    learning_rate        | 0.0003      |
|    loss                 | -0.29       |
|    n_updates            | 180         |
|    policy_gradient_loss | -0.00727    |
|    std                  | 1.82        |
|    value_loss           | 0.00143     |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 4           |
|    iterations           | 20          |
|    time_elapsed         | 8831        |
|    total_timesteps      | 40960       |
| train/                  |             |
|    approx_kl            | 0.011269572 |
|    clip_fraction        | 0.0961      |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.2       |
|    explained_variance   | 0.478       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.266      |
|    n_updates            | 190         |
|    policy_gradient_loss | -0.0051     |
|    std                  | 1.83        |
|    value_loss           | 0.00146     |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 4           |
|    iterations           | 21          |
|    time_elapsed         | 9026        |
|    total_timesteps      | 43008       |
| train/                  |             |
|    approx_kl            | 0.011720242 |
|    clip_fraction        | 0.116       |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.3       |
|    explained_variance   | 0.403       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.276      |
|    n_updates            | 200         |
|    policy_gradient_loss | -0.00566    |
|    std                  | 1.83        |
|    value_loss           | 0.00127     |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 4           |
|    iterations           | 22          |
|    time_elapsed         | 9219        |
|    total_timesteps      | 45056       |
| train/                  |             |
|    approx_kl            | 0.010057259 |
|    clip_fraction        | 0.11        |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.3       |
|    explained_variance   | 0.438       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.292      |
|    n_updates            | 210         |
|    policy_gradient_loss | -0.00873    |
|    std                  | 1.84        |
|    value_loss           | 0.001       |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 5           |
|    iterations           | 23          |
|    time_elapsed         | 9414        |
|    total_timesteps      | 47104       |
| train/                  |             |
|    approx_kl            | 0.009345452 |
|    clip_fraction        | 0.0958      |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.3       |
|    explained_variance   | 0.504       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.273      |
|    n_updates            | 220         |
|    policy_gradient_loss | -0.00719    |
|    std                  | 1.84        |
|    value_loss           | 0.00101     |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 5           |
|    iterations           | 24          |
|    time_elapsed         | 9607        |
|    total_timesteps      | 49152       |
| train/                  |             |
|    approx_kl            | 0.008346567 |
|    clip_fraction        | 0.0878      |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.4       |
|    explained_variance   | 0.475       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.28       |
|    n_updates            | 230         |
|    policy_gradient_loss | -0.00692    |
|    std                  | 1.84        |
|    value_loss           | 0.00116     |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 5           |
|    iterations           | 25          |
|    time_elapsed         | 9800        |
|    total_timesteps      | 51200       |
| train/                  |             |
|    approx_kl            | 0.011436922 |
|    clip_fraction        | 0.113       |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.3       |
|    explained_variance   | 0.401       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.285      |
|    n_updates            | 240         |
|    policy_gradient_loss | -0.00983    |
|    std                  | 1.83        |
|    value_loss           | 0.00123     |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 5           |
|    iterations           | 26          |
|    time_elapsed         | 9995        |
|    total_timesteps      | 53248       |
| train/                  |             |
|    approx_kl            | 0.009876674 |
|    clip_fraction        | 0.103       |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.3       |
|    explained_variance   | 0.41        |
|    learning_rate        | 0.0003      |
|    loss                 | -0.28       |
|    n_updates            | 250         |
|    policy_gradient_loss | -0.00912    |
|    std                  | 1.83        |
|    value_loss           | 0.00109     |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 5           |
|    iterations           | 27          |
|    time_elapsed         | 10187       |
|    total_timesteps      | 55296       |
| train/                  |             |
|    approx_kl            | 0.014652755 |
|    clip_fraction        | 0.151       |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.4       |
|    explained_variance   | 0.394       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.27       |
|    n_updates            | 260         |
|    policy_gradient_loss | -0.012      |
|    std                  | 1.85        |
|    value_loss           | 0.000809    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 5           |
|    iterations           | 28          |
|    time_elapsed         | 10381       |
|    total_timesteps      | 57344       |
| train/                  |             |
|    approx_kl            | 0.012147739 |
|    clip_fraction        | 0.136       |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.4       |
|    explained_variance   | 0.456       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.285      |
|    n_updates            | 270         |
|    policy_gradient_loss | -0.0127     |
|    std                  | 1.85        |
|    value_loss           | 0.000872    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 5           |
|    iterations           | 29          |
|    time_elapsed         | 10574       |
|    total_timesteps      | 59392       |
| train/                  |             |
|    approx_kl            | 0.020403909 |
|    clip_fraction        | 0.138       |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.5       |
|    explained_variance   | 0.445       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.295      |
|    n_updates            | 280         |
|    policy_gradient_loss | -0.013      |
|    std                  | 1.86        |
|    value_loss           | 0.00103     |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 5           |
|    iterations           | 30          |
|    time_elapsed         | 10769       |
|    total_timesteps      | 61440       |
| train/                  |             |
|    approx_kl            | 0.014676759 |
|    clip_fraction        | 0.148       |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.5       |
|    explained_variance   | 0.401       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.289      |
|    n_updates            | 290         |
|    policy_gradient_loss | -0.0124     |
|    std                  | 1.86        |
|    value_loss           | 0.000804    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 5           |
|    iterations           | 31          |
|    time_elapsed         | 10968       |
|    total_timesteps      | 63488       |
| train/                  |             |
|    approx_kl            | 0.014184393 |
|    clip_fraction        | 0.148       |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.6       |
|    explained_variance   | 0.421       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.26       |
|    n_updates            | 300         |
|    policy_gradient_loss | -0.0106     |
|    std                  | 1.88        |
|    value_loss           | 0.000944    |
-----------------------------------------
----------------------------------------
| time/                   |            |
|    fps                  | 5          |
|    iterations           | 32         |
|    time_elapsed         | 11166      |
|    total_timesteps      | 65536      |
| train/                  |            |
|    approx_kl            | 0.01339736 |
|    clip_fraction        | 0.122      |
|    clip_range           | 0.2        |
|    entropy_loss         | -26.6      |
|    explained_variance   | 0.409      |
|    learning_rate        | 0.0003     |
|    loss                 | -0.284     |
|    n_updates            | 310        |
|    policy_gradient_loss | -0.00947   |
|    std                  | 1.88       |
|    value_loss           | 0.000944   |
----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 5           |
|    iterations           | 33          |
|    time_elapsed         | 11362       |
|    total_timesteps      | 67584       |
| train/                  |             |
|    approx_kl            | 0.020863561 |
|    clip_fraction        | 0.176       |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.7       |
|    explained_variance   | 0.405       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.279      |
|    n_updates            | 320         |
|    policy_gradient_loss | -0.0144     |
|    std                  | 1.89        |
|    value_loss           | 0.000908    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 6           |
|    iterations           | 34          |
|    time_elapsed         | 11557       |
|    total_timesteps      | 69632       |
| train/                  |             |
|    approx_kl            | 0.019992603 |
|    clip_fraction        | 0.171       |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.7       |
|    explained_variance   | 0.432       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.299      |
|    n_updates            | 330         |
|    policy_gradient_loss | -0.0131     |
|    std                  | 1.89        |
|    value_loss           | 0.000952    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 6           |
|    iterations           | 35          |
|    time_elapsed         | 11750       |
|    total_timesteps      | 71680       |
| train/                  |             |
|    approx_kl            | 0.015240448 |
|    clip_fraction        | 0.141       |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.7       |
|    explained_variance   | 0.458       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.286      |
|    n_updates            | 340         |
|    policy_gradient_loss | -0.00952    |
|    std                  | 1.89        |
|    value_loss           | 0.000753    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 6           |
|    iterations           | 36          |
|    time_elapsed         | 11945       |
|    total_timesteps      | 73728       |
| train/                  |             |
|    approx_kl            | 0.017300546 |
|    clip_fraction        | 0.157       |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.7       |
|    explained_variance   | 0.503       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.282      |
|    n_updates            | 350         |
|    policy_gradient_loss | -0.0109     |
|    std                  | 1.9         |
|    value_loss           | 0.000866    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 6           |
|    iterations           | 37          |
|    time_elapsed         | 12138       |
|    total_timesteps      | 75776       |
| train/                  |             |
|    approx_kl            | 0.014761126 |
|    clip_fraction        | 0.152       |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.7       |
|    explained_variance   | 0.486       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.296      |
|    n_updates            | 360         |
|    policy_gradient_loss | -0.01       |
|    std                  | 1.89        |
|    value_loss           | 0.000933    |
-----------------------------------------
----------------------------------------
| time/                   |            |
|    fps                  | 6          |
|    iterations           | 38         |
|    time_elapsed         | 12331      |
|    total_timesteps      | 77824      |
| train/                  |            |
|    approx_kl            | 0.02182821 |
|    clip_fraction        | 0.158      |
|    clip_range           | 0.2        |
|    entropy_loss         | -26.7      |
|    explained_variance   | 0.457      |
|    learning_rate        | 0.0003     |
|    loss                 | -0.321     |
|    n_updates            | 370        |
|    policy_gradient_loss | -0.0117    |
|    std                  | 1.89       |
|    value_loss           | 0.00079    |
----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 6           |
|    iterations           | 39          |
|    time_elapsed         | 12526       |
|    total_timesteps      | 79872       |
| train/                  |             |
|    approx_kl            | 0.020453628 |
|    clip_fraction        | 0.157       |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.7       |
|    explained_variance   | 0.499       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.278      |
|    n_updates            | 380         |
|    policy_gradient_loss | -0.00984    |
|    std                  | 1.89        |
|    value_loss           | 0.000926    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 6           |
|    iterations           | 40          |
|    time_elapsed         | 12719       |
|    total_timesteps      | 81920       |
| train/                  |             |
|    approx_kl            | 0.027308898 |
|    clip_fraction        | 0.176       |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.8       |
|    explained_variance   | 0.454       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.326      |
|    n_updates            | 390         |
|    policy_gradient_loss | -0.0153     |
|    std                  | 1.9         |
|    value_loss           | 0.000854    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 6           |
|    iterations           | 41          |
|    time_elapsed         | 12914       |
|    total_timesteps      | 83968       |
| train/                  |             |
|    approx_kl            | 0.017341875 |
|    clip_fraction        | 0.177       |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.8       |
|    explained_variance   | 0.421       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.288      |
|    n_updates            | 400         |
|    policy_gradient_loss | -0.0114     |
|    std                  | 1.91        |
|    value_loss           | 0.00119     |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 6           |
|    iterations           | 42          |
|    time_elapsed         | 13107       |
|    total_timesteps      | 86016       |
| train/                  |             |
|    approx_kl            | 0.023058547 |
|    clip_fraction        | 0.158       |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.9       |
|    explained_variance   | 0.5         |
|    learning_rate        | 0.0003      |
|    loss                 | -0.271      |
|    n_updates            | 410         |
|    policy_gradient_loss | -0.0103     |
|    std                  | 1.91        |
|    value_loss           | 0.000967    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 6           |
|    iterations           | 43          |
|    time_elapsed         | 13301       |
|    total_timesteps      | 88064       |
| train/                  |             |
|    approx_kl            | 0.021509882 |
|    clip_fraction        | 0.179       |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.9       |
|    explained_variance   | 0.453       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.304      |
|    n_updates            | 420         |
|    policy_gradient_loss | -0.0133     |
|    std                  | 1.92        |
|    value_loss           | 0.0009      |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 6           |
|    iterations           | 44          |
|    time_elapsed         | 13495       |
|    total_timesteps      | 90112       |
| train/                  |             |
|    approx_kl            | 0.022018932 |
|    clip_fraction        | 0.179       |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.9       |
|    explained_variance   | 0.496       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.286      |
|    n_updates            | 430         |
|    policy_gradient_loss | -0.013      |
|    std                  | 1.92        |
|    value_loss           | 0.00103     |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 6           |
|    iterations           | 45          |
|    time_elapsed         | 13687       |
|    total_timesteps      | 92160       |
| train/                  |             |
|    approx_kl            | 0.020589344 |
|    clip_fraction        | 0.178       |
|    clip_range           | 0.2         |
|    entropy_loss         | -26.9       |
|    explained_variance   | 0.484       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.293      |
|    n_updates            | 440         |
|    policy_gradient_loss | -0.0157     |
|    std                  | 1.93        |
|    value_loss           | 0.000918    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 6           |
|    iterations           | 46          |
|    time_elapsed         | 13882       |
|    total_timesteps      | 94208       |
| train/                  |             |
|    approx_kl            | 0.024633601 |
|    clip_fraction        | 0.201       |
|    clip_range           | 0.2         |
|    entropy_loss         | -27         |
|    explained_variance   | 0.533       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.302      |
|    n_updates            | 450         |
|    policy_gradient_loss | -0.0151     |
|    std                  | 1.94        |
|    value_loss           | 0.000798    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 6           |
|    iterations           | 47          |
|    time_elapsed         | 14076       |
|    total_timesteps      | 96256       |
| train/                  |             |
|    approx_kl            | 0.018570691 |
|    clip_fraction        | 0.162       |
|    clip_range           | 0.2         |
|    entropy_loss         | -27.1       |
|    explained_variance   | 0.496       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.275      |
|    n_updates            | 460         |
|    policy_gradient_loss | -0.0139     |
|    std                  | 1.95        |
|    value_loss           | 0.000948    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 6           |
|    iterations           | 48          |
|    time_elapsed         | 14271       |
|    total_timesteps      | 98304       |
| train/                  |             |
|    approx_kl            | 0.016717477 |
|    clip_fraction        | 0.165       |
|    clip_range           | 0.2         |
|    entropy_loss         | -27.1       |
|    explained_variance   | 0.45        |
|    learning_rate        | 0.0003      |
|    loss                 | -0.268      |
|    n_updates            | 470         |
|    policy_gradient_loss | -0.0135     |
|    std                  | 1.95        |
|    value_loss           | 0.00111     |
-----------------------------------------
----------------------------------------
| time/                   |            |
|    fps                  | 6          |
|    iterations           | 49         |
|    time_elapsed         | 14463      |
|    total_timesteps      | 100352     |
| train/                  |            |
|    approx_kl            | 0.04168821 |
|    clip_fraction        | 0.186      |
|    clip_range           | 0.2        |
|    entropy_loss         | -27.2      |
|    explained_variance   | 0.547      |
|    learning_rate        | 0.0003     |
|    loss                 | -0.297     |
|    n_updates            | 480        |
|    policy_gradient_loss | -0.0154    |
|    std                  | 1.97       |
|    value_loss           | 0.000899   |
----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 6           |
|    iterations           | 50          |
|    time_elapsed         | 14653       |
|    total_timesteps      | 102400      |
| train/                  |             |
|    approx_kl            | 0.022852132 |
|    clip_fraction        | 0.203       |
|    clip_range           | 0.2         |
|    entropy_loss         | -27.3       |
|    explained_variance   | 0.505       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.268      |
|    n_updates            | 490         |
|    policy_gradient_loss | -0.0156     |
|    std                  | 1.98        |
|    value_loss           | 0.000904    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 7           |
|    iterations           | 51          |
|    time_elapsed         | 14845       |
|    total_timesteps      | 104448      |
| train/                  |             |
|    approx_kl            | 0.018193016 |
|    clip_fraction        | 0.148       |
|    clip_range           | 0.2         |
|    entropy_loss         | -27.3       |
|    explained_variance   | 0.535       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.254      |
|    n_updates            | 500         |
|    policy_gradient_loss | -0.0113     |
|    std                  | 1.99        |
|    value_loss           | 0.000819    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 7           |
|    iterations           | 52          |
|    time_elapsed         | 15045       |
|    total_timesteps      | 106496      |
| train/                  |             |
|    approx_kl            | 0.019693835 |
|    clip_fraction        | 0.17        |
|    clip_range           | 0.2         |
|    entropy_loss         | -27.4       |
|    explained_variance   | 0.591       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.303      |
|    n_updates            | 510         |
|    policy_gradient_loss | -0.0133     |
|    std                  | 2           |
|    value_loss           | 0.000922    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 7           |
|    iterations           | 53          |
|    time_elapsed         | 15236       |
|    total_timesteps      | 108544      |
| train/                  |             |
|    approx_kl            | 0.020735016 |
|    clip_fraction        | 0.179       |
|    clip_range           | 0.2         |
|    entropy_loss         | -27.5       |
|    explained_variance   | 0.584       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.32       |
|    n_updates            | 520         |
|    policy_gradient_loss | -0.0153     |
|    std                  | 2.01        |
|    value_loss           | 0.000863    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 7           |
|    iterations           | 54          |
|    time_elapsed         | 15429       |
|    total_timesteps      | 110592      |
| train/                  |             |
|    approx_kl            | 0.019577226 |
|    clip_fraction        | 0.188       |
|    clip_range           | 0.2         |
|    entropy_loss         | -27.5       |
|    explained_variance   | 0.613       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.302      |
|    n_updates            | 530         |
|    policy_gradient_loss | -0.0161     |
|    std                  | 2.02        |
|    value_loss           | 0.000772    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 7           |
|    iterations           | 55          |
|    time_elapsed         | 15624       |
|    total_timesteps      | 112640      |
| train/                  |             |
|    approx_kl            | 0.020303361 |
|    clip_fraction        | 0.186       |
|    clip_range           | 0.2         |
|    entropy_loss         | -27.5       |
|    explained_variance   | 0.634       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.291      |
|    n_updates            | 540         |
|    policy_gradient_loss | -0.0164     |
|    std                  | 2.02        |
|    value_loss           | 0.000896    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 7           |
|    iterations           | 56          |
|    time_elapsed         | 15816       |
|    total_timesteps      | 114688      |
| train/                  |             |
|    approx_kl            | 0.016390374 |
|    clip_fraction        | 0.174       |
|    clip_range           | 0.2         |
|    entropy_loss         | -27.6       |
|    explained_variance   | 0.563       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.3        |
|    n_updates            | 550         |
|    policy_gradient_loss | -0.0132     |
|    std                  | 2.02        |
|    value_loss           | 0.000984    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 7           |
|    iterations           | 57          |
|    time_elapsed         | 16012       |
|    total_timesteps      | 116736      |
| train/                  |             |
|    approx_kl            | 0.022534795 |
|    clip_fraction        | 0.211       |
|    clip_range           | 0.2         |
|    entropy_loss         | -27.6       |
|    explained_variance   | 0.64        |
|    learning_rate        | 0.0003      |
|    loss                 | -0.283      |
|    n_updates            | 560         |
|    policy_gradient_loss | -0.0164     |
|    std                  | 2.03        |
|    value_loss           | 0.000858    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 7           |
|    iterations           | 58          |
|    time_elapsed         | 16204       |
|    total_timesteps      | 118784      |
| train/                  |             |
|    approx_kl            | 0.022122987 |
|    clip_fraction        | 0.213       |
|    clip_range           | 0.2         |
|    entropy_loss         | -27.6       |
|    explained_variance   | 0.557       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.297      |
|    n_updates            | 570         |
|    policy_gradient_loss | -0.0147     |
|    std                  | 2.03        |
|    value_loss           | 0.00074     |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 7           |
|    iterations           | 59          |
|    time_elapsed         | 16465       |
|    total_timesteps      | 120832      |
| train/                  |             |
|    approx_kl            | 0.017415296 |
|    clip_fraction        | 0.183       |
|    clip_range           | 0.2         |
|    entropy_loss         | -27.7       |
|    explained_variance   | 0.594       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.275      |
|    n_updates            | 580         |
|    policy_gradient_loss | -0.0121     |
|    std                  | 2.04        |
|    value_loss           | 0.000817    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 7           |
|    iterations           | 60          |
|    time_elapsed         | 16948       |
|    total_timesteps      | 122880      |
| train/                  |             |
|    approx_kl            | 0.022651702 |
|    clip_fraction        | 0.229       |
|    clip_range           | 0.2         |
|    entropy_loss         | -27.7       |
|    explained_variance   | 0.602       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.342      |
|    n_updates            | 590         |
|    policy_gradient_loss | -0.0162     |
|    std                  | 2.04        |
|    value_loss           | 0.000839    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 3           |
|    iterations           | 61          |
|    time_elapsed         | 34183       |
|    total_timesteps      | 124928      |
| train/                  |             |
|    approx_kl            | 0.023445208 |
|    clip_fraction        | 0.206       |
|    clip_range           | 0.2         |
|    entropy_loss         | -27.7       |
|    explained_variance   | 0.569       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.279      |
|    n_updates            | 600         |
|    policy_gradient_loss | -0.0148     |
|    std                  | 2.04        |
|    value_loss           | 0.000953    |
-----------------------------------------
----------------------------------------
| time/                   |            |
|    fps                  | 3          |
|    iterations           | 62         |
|    time_elapsed         | 35892      |
|    total_timesteps      | 126976     |
| train/                  |            |
|    approx_kl            | 0.02599547 |
|    clip_fraction        | 0.204      |
|    clip_range           | 0.2        |
|    entropy_loss         | -27.7      |
|    explained_variance   | 0.576      |
|    learning_rate        | 0.0003     |
|    loss                 | -0.298     |
|    n_updates            | 610        |
|    policy_gradient_loss | -0.0123    |
|    std                  | 2.05       |
|    value_loss           | 0.000899   |
----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 3           |
|    iterations           | 63          |
|    time_elapsed         | 39037       |
|    total_timesteps      | 129024      |
| train/                  |             |
|    approx_kl            | 0.027264554 |
|    clip_fraction        | 0.206       |
|    clip_range           | 0.2         |
|    entropy_loss         | -27.7       |
|    explained_variance   | 0.597       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.292      |
|    n_updates            | 620         |
|    policy_gradient_loss | -0.0152     |
|    std                  | 2.05        |
|    value_loss           | 0.000697    |
-----------------------------------------
----------------------------------------
| time/                   |            |
|    fps                  | 3          |
|    iterations           | 64         |
|    time_elapsed         | 39249      |
|    total_timesteps      | 131072     |
| train/                  |            |
|    approx_kl            | 0.03108087 |
|    clip_fraction        | 0.239      |
|    clip_range           | 0.2        |
|    entropy_loss         | -27.8      |
|    explained_variance   | 0.518      |
|    learning_rate        | 0.0003     |
|    loss                 | -0.279     |
|    n_updates            | 630        |
|    policy_gradient_loss | -0.0139    |
|    std                  | 2.06       |
|    value_loss           | 0.000813   |
----------------------------------------
----------------------------------------
| time/                   |            |
|    fps                  | 3          |
|    iterations           | 65         |
|    time_elapsed         | 39446      |
|    total_timesteps      | 133120     |
| train/                  |            |
|    approx_kl            | 0.02868305 |
|    clip_fraction        | 0.21       |
|    clip_range           | 0.2        |
|    entropy_loss         | -27.9      |
|    explained_variance   | 0.577      |
|    learning_rate        | 0.0003     |
|    loss                 | -0.269     |
|    n_updates            | 640        |
|    policy_gradient_loss | -0.00835   |
|    std                  | 2.08       |
|    value_loss           | 0.000824   |
----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 3           |
|    iterations           | 66          |
|    time_elapsed         | 39649       |
|    total_timesteps      | 135168      |
| train/                  |             |
|    approx_kl            | 0.025583021 |
|    clip_fraction        | 0.21        |
|    clip_range           | 0.2         |
|    entropy_loss         | -27.9       |
|    explained_variance   | 0.518       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.268      |
|    n_updates            | 650         |
|    policy_gradient_loss | -0.0122     |
|    std                  | 2.08        |
|    value_loss           | 0.000959    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 3           |
|    iterations           | 67          |
|    time_elapsed         | 39856       |
|    total_timesteps      | 137216      |
| train/                  |             |
|    approx_kl            | 0.023820087 |
|    clip_fraction        | 0.22        |
|    clip_range           | 0.2         |
|    entropy_loss         | -28         |
|    explained_variance   | 0.597       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.278      |
|    n_updates            | 660         |
|    policy_gradient_loss | -0.0126     |
|    std                  | 2.1         |
|    value_loss           | 0.000853    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 3           |
|    iterations           | 68          |
|    time_elapsed         | 40066       |
|    total_timesteps      | 139264      |
| train/                  |             |
|    approx_kl            | 0.026507173 |
|    clip_fraction        | 0.207       |
|    clip_range           | 0.2         |
|    entropy_loss         | -28         |
|    explained_variance   | 0.595       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.31       |
|    n_updates            | 670         |
|    policy_gradient_loss | -0.0119     |
|    std                  | 2.1         |
|    value_loss           | 0.000807    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 3           |
|    iterations           | 69          |
|    time_elapsed         | 40273       |
|    total_timesteps      | 141312      |
| train/                  |             |
|    approx_kl            | 0.021159166 |
|    clip_fraction        | 0.21        |
|    clip_range           | 0.2         |
|    entropy_loss         | -28.1       |
|    explained_variance   | 0.534       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.283      |
|    n_updates            | 680         |
|    policy_gradient_loss | -0.0145     |
|    std                  | 2.1         |
|    value_loss           | 0.000835    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 3           |
|    iterations           | 70          |
|    time_elapsed         | 40484       |
|    total_timesteps      | 143360      |
| train/                  |             |
|    approx_kl            | 0.022724248 |
|    clip_fraction        | 0.221       |
|    clip_range           | 0.2         |
|    entropy_loss         | -28.1       |
|    explained_variance   | 0.594       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.293      |
|    n_updates            | 690         |
|    policy_gradient_loss | -0.0145     |
|    std                  | 2.11        |
|    value_loss           | 0.000834    |
-----------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 3           |
|    iterations           | 71          |
|    time_elapsed         | 40696       |
|    total_timesteps      | 145408      |
| train/                  |             |
|    approx_kl            | 0.030216534 |
|    clip_fraction        | 0.234       |
|    clip_range           | 0.2         |
|    entropy_loss         | -28.2       |
|    explained_variance   | 0.547       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.314      |
|    n_updates            | 700         |
|    policy_gradient_loss | -0.0158     |
|    std                  | 2.12        |
|    value_loss           | 0.000899    |
-----------------------------------------
---------------------------------------
| time/                   |           |
|    fps                  | 3         |
|    iterations           | 72        |
|    time_elapsed         | 40904     |
|    total_timesteps      | 147456    |
| train/                  |           |
|    approx_kl            | 0.0289419 |
|    clip_fraction        | 0.21      |
|    clip_range           | 0.2       |
|    entropy_loss         | -28.2     |
|    explained_variance   | 0.485     |
|    learning_rate        | 0.0003    |
|    loss                 | -0.306    |
|    n_updates            | 710       |
|    policy_gradient_loss | -0.0117   |
|    std                  | 2.13      |
|    value_loss           | 0.00102   |
---------------------------------------
-----------------------------------------
| time/                   |             |
|    fps                  | 3           |
|    iterations           | 73          |
|    time_elapsed         | 41114       |
|    total_timesteps      | 149504      |
| train/                  |             |
|    approx_kl            | 0.023294847 |
|    clip_fraction        | 0.207       |
|    clip_range           | 0.2         |
|    entropy_loss         | -28.3       |
|    explained_variance   | 0.513       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.287      |
|    n_updates            | 720         |
|    policy_gradient_loss | -0.00879    |
|    std                  | 2.14        |
|    value_loss           | 0.0011      |
-----------------------------------------
----------------------------------------
| time/                   |            |
|    fps                  | 3          |
|    iterations           | 74         |
|    time_elapsed         | 41394      |
|    total_timesteps      | 151552     |
| train/                  |            |
|    approx_kl            | 0.03114854 |
|    clip_fraction        | 0.225      |
|    clip_range           | 0.2        |
|    entropy_loss         | -28.3      |
|    explained_variance   | 0.516      |
|    learning_rate        | 0.0003     |
|    loss                 | -0.308     |
|    n_updates            | 730        |
|    policy_gradient_loss | -0.0154    |
|    std                  | 2.13       |
|    value_loss           | 0.00102    |
----------------------------------------
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 151,552/150,000  [ 4:48:39 < 0:00:00 , ? it/s ]

✅ 학습 완료! 모델 저장: results/ppo_portfolio_final.zip

📊 에이전트 평가 중 (3 에피소드)...
  Episode 1: Return=108.03%, Sharpe=1.17, MDD=-16.56%
  Episode 2: Return=108.03%, Sharpe=1.17, MDD=-16.56%
  Episode 3: Return=108.03%, Sharpe=1.17, MDD=-16.56%

✅ 평가 완료 (평균):
   Total Return: 108.03%
   Annualized Return: 51.55%
   Sharpe Ratio: 1.17
   Max Drawdown: -16.56%

🔍 베이스라인 전략 평가: equal_weight
  Return: 34.45%, Sharpe: 1.02, MDD: -9.66%

🔍 베이스라인 전략 평가: buy_and_hold
  Return: 34.45%, Sharpe: 1.02, MDD: -9.66%

📊 결과 시각화 중...
📈 차트 저장: results/rl_performance.png
📊 비교 차트 저장: results/strategy_comparison.png

💾 결과 저장 완료:
   - results/metrics.csv
   - results/capital_history.csv

======================================================================
✅ 모든 작업 완료!
======================================================================

📁 결과물:
   - 모델: results/ppo_portfolio_final.zip
   - 성과 차트: results/rl_performance.png
   - 전략 비교: results/strategy_comparison.png
   - 메트릭: results/metrics.csv
   - 자본 이력: results/capital_history.csv

💡 TensorBoard 확인:
   tensorboard --logdir results/tensorboard

🔮 미래 테스트 (제출 후):
   2025년 12월 데이터로 최종 검증:
   1. python src/collect_december.py (12월 후)
   2. python src/evaluate_december.py

📊 프로젝트 성과 요약:
   기간: 2024.01 ~ 2025.11 (23개월)
   총 수익률: 108.03%
   연환산 수익률: 51.55%
   Sharpe Ratio: 1.17
   Max Drawdown: -16.56%


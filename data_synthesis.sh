#!/bin/bash

# 用 venv 的 python(pyenv shim 沒裝 numpy 等套件)
PYTHON='/Users/rex/covsyn/bin/python'

cpu_cores=24
# B36: the Taiwan first-wave scenario is kept as an end-to-end demonstration (it is the only
# output compared with a real epidemic curve), so it is generated alongside the single-seed
# spread data that every validation figure uses.
modes=('spread_Taiwan_weight' 'taiwan_first_outbreak')

# 指向「剛跑完的新訓練結果」(專案根目錄那個,20-39 鎖 1 的版本)。
parameter_path='./Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_200'
# A5: at least 1000 Monte Carlo runs; 100 was shown to be unreliable.
monte_carlo_number=1000

# 先確認最佳參數檔存在,否則直接報錯比較好 debug。
if [ ! -f "$parameter_path/firefly_best.txt" ]; then
    echo "ERROR: 找不到 $parameter_path/firefly_best.txt"
    echo "請確認 parameter_path 指向正確的訓練結果資料夾。"
    exit 1
fi

# if result_path does not exist, create it
for mode in "${modes[@]}"; do
    result_path="./synthetic_data_results_$mode"
    mkdir -p "$result_path"

    "$PYTHON" ./Data_synthesis_main.py --mode "$mode" \
        --monte_carlo_number "$monte_carlo_number" \
        --result_path "$result_path" \
        --cpu_cores "$cpu_cores" \
        --parameter_path "$parameter_path"

    # Print out the result_path
    echo "Done! Results are saved in: $result_path"
done

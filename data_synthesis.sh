#!/bin/bash

cpu_cores=16
modes=('spread_Taiwan_weight' 'cheng2020' 'taiwan_first_outbreak')
# modes=('taiwan_first_outbreak')

parameter_path='./firefly_result/Firefly_result_pop_size_100_alpha_1_betamin_1_gamma_0.131_max_generations_200'
monte_carlo_number=1100

# if result_path does not exist, create it
for mode in "${modes[@]}"; do
    result_path="./synthetic_data_results_$mode"
    if [ ! -d "$result_path" ]; then
        mkdir "$result_path"
    fi
    
    python ./Data_synthesis_main.py --mode "$mode" \
        --monte_carlo_number "$monte_carlo_number" \
        --result_path "$result_path" \
        --cpu_cores "$cpu_cores" \
        --parameter_path "$parameter_path"

    # Print out the result_path
    echo "Done! Results are saved in: $result_path"
done

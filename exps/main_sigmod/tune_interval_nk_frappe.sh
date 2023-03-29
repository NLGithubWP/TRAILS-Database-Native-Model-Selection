


dataset=frappe

for((num_points=8; num_points <= 9; ++num_points)); do
  for((kn_rate=50; kn_rate <= 60; kn_rate+=10)); do

    python ./exps/main_sigmod/system/simulate.py \
      --dataset=frappe \
      --kn_rate=$kn_rate \
      --num_points=$num_points

    file_name="./exps/main_sigmod/analysis/res_end_2_end_$dataset""_$kn_rate""_$num_points.json"
    python ./exps/main_sigmod/analysis/simple_draw.py \
      --saved_result=$file_name

  done
done
#!/usr/bin/zsh
#
# Usage: ./grid_search.sh
#
# This script performs a grid search across multiple datasets, modes, clustering methods,
# and normalization techniques for the QuestEA project. It runs the QuestEA.py script
# with various combinations of parameters, processes the results, and generates plots.
#
# The script also:
# - Sends notifications to a phone at key points in the process
# - Manages system swap to ensure optimal performance
# - Cleans up intermediate results
# - Opens result visualizations upon completion

# send notification to phone
function phone_notif() {
    sender="laptop_tFzfJNlDSjNFA"
    title="PROJ"
    message="$1"
    #curl -H "Title: $title" -d "$message" "ntfy.sh/$sender"
}

function call() {
    phone_notif "START $@"
    python $@
}


function main() {

    norm_list=("l2" "l1")
    dataset_list=("16PF" "DASS" "Hamilton" "HEXACO" "IPIP")
    method_list=("kmeans")
    #method_list=("kmeans" "SpectralCosine")
    #norm_list=("l2")
    mode_list=("default_raw" "default_agg" "sbert" "openai")

    # counting number of iteration
    total=0
    for dataset in "${dataset_list[@]}" ; do
        for mode in "${mode_list[@]}" ; do
            for method in "${method_list[@]}" ; do
                for norm in "${norm_list[@]}" ; do
                    total=$(($total+1))
                done
            done
        done
    done

    rm -rv results_ignore_backups

    cnt=0
    for dataset in "${dataset_list[@]}"
    do
        for mode in "${mode_list[@]}"
        do
            for method in "${method_list[@]}"
            do
                for norm in "${norm_list[@]}"
                do
                    sudo swapoff -av && sudo swapon -av
                    call ./QuestEA.py \
                        --mode=$mode \
                        --note=$mode \
                        --cluster_method=$method \
                        --norm=$norm \
                        --n_cpus=$(cat n_cpus) \
                        --n_components=20 \
                        --sample_to_keep=10000 \
                        --datasetname=$dataset
                    cnt=$(($cnt+1)) ; echo $cnt| tqdm --total $total --update_to --desc "BATCH PROGRESS" --null
                    echo ; echo ; echo ; echo ; echo
                    sleep 2
                done
            done
        done
        rm -rv "results_ignore_backups/$dataset/batch_cache"
        phone_notif "computing results"
        python compare_results.py --paths="./results_ignore_backups/$dataset" --behavior="store"
        python plotter.py --open_plot=False --paths="./results_ignore_backups/$dataset" --behavior="store"
    done

    sudo swapoff -av && sudo swapon -av

    find results_ignore_backups/ -name "network.png" | xargs -I% -P 10 xdg-open "%"

    echo "FINISHED"
}

(main || phone_notif "Error during script")

#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 [rl] [fields2cover] [distributions] [prior_knowledge_quality] [detection_errors] [stopping_criteria]"
    exit 1
fi

contains_option() {
    local target="$1"
    shift
    
    for arg in "$@"; do
        if [ "$arg" == "$target" ]; then
            return 0 
        fi
    done
    
    return 1 
}

if contains_option "distributions" "$@"; then
    if contains_option "rl" "$@"; then
        gnome-terminal -- python3 evaluate.py rl --output_folder evaluations/distributions --prefix distribution_random --n_episodes 1000 --env_args config_file=experiments/distributions/env_config_random.yaml --weight_file training/distributions/distribution_random_best.pt --seed 1234 --video
        gnome-terminal -- python3 evaluate.py rl --output_folder evaluations/distributions --prefix distribution_medium --n_episodes 1000 --env_args config_file=experiments/distributions/env_config_medium.yaml --weight_file training/distributions/distribution_medium_best.pt --seed 1234 --video
        gnome-terminal -- python3 evaluate.py rl --output_folder evaluations/distributions --prefix distribution_strong --n_episodes 1000 --env_args config_file=experiments/distributions/env_config_strong.yaml --weight_file training/distributions/distribution_strong_best.pt --seed 1234 --video
    fi
    if contains_option "fields2cover" "$@"; then
        gnome-terminal -- python3 evaluate.py fields2cover --output_folder evaluations/distributions --prefix distribution_random --n_episodes 1000 --env_args config_file=experiments/distributions/env_config_random.yaml --seed 1234 --video
        gnome-terminal -- python3 evaluate.py fields2cover --output_folder evaluations/distributions --prefix distribution_medium --n_episodes 1000 --env_args config_file=experiments/distributions/env_config_medium.yaml --seed 1234 --video
        gnome-terminal -- python3 evaluate.py fields2cover --output_folder evaluations/distributions --prefix distribution_strong --n_episodes 1000 --env_args config_file=experiments/distributions/env_config_strong.yaml --seed 1234 --video
    fi
fi

if contains_option "prior_knowledge_quality" "$@"; then
    if contains_option "rl" "$@"; then
        gnome-terminal -- python3 evaluate.py rl --output_folder evaluations/prior_knowledge_quality --prefix prior_knowledge_uncertainty_level_0 --n_episodes 1000 --env_args config_file=experiments/prior_knowledge_quality/env_config_level_0.yaml --weight_file training/prior_knowledge_quality/prior_knowledge_uncertainty_level_0_best.pt --seed 1234 --video
        gnome-terminal -- python3 evaluate.py rl --output_folder evaluations/prior_knowledge_quality --prefix prior_knowledge_uncertainty_level_1 --n_episodes 1000 --env_args config_file=experiments/prior_knowledge_quality/env_config_level_1.yaml --weight_file training/prior_knowledge_quality/prior_knowledge_uncertainty_level_1_best.pt --seed 1234 --video
        gnome-terminal -- python3 evaluate.py rl --output_folder evaluations/prior_knowledge_quality --prefix prior_knowledge_uncertainty_level_2 --n_episodes 1000 --env_args config_file=experiments/prior_knowledge_quality/env_config_level_2.yaml --weight_file training/prior_knowledge_quality/prior_knowledge_uncertainty_level_2_best.pt --seed 1234 --video
        gnome-terminal -- python3 evaluate.py rl --output_folder evaluations/prior_knowledge_quality --prefix prior_knowledge_uncertainty_level_3 --n_episodes 1000 --env_args config_file=experiments/prior_knowledge_quality/env_config_level_3.yaml --weight_file training/prior_knowledge_quality/prior_knowledge_uncertainty_level_3_best.pt --seed 1234 --video
        gnome-terminal -- python3 evaluate.py rl --output_folder evaluations/prior_knowledge_quality --prefix prior_knowledge_uncertainty_level_4 --n_episodes 1000 --env_args config_file=experiments/prior_knowledge_quality/env_config_level_4.yaml --weight_file training/prior_knowledge_quality/prior_knowledge_uncertainty_level_4_best.pt --seed 1234 --video
    fi
    if contains_option "fields2cover" "$@"; then
        gnome-terminal -- python3 evaluate.py fields2cover --output_folder evaluations/prior_knowledge_quality --prefix prior_knowledge_uncertainty --n_episodes 1000 --env_args config_file=experiments/prior_knowledge_quality/env_config_level_2.yaml --seed 1234 --video
    fi
fi

if contains_option "detection_errors" "$@"; then
    if contains_option "rl" "$@"; then
        gnome-terminal -- python3 evaluate.py rl --output_folder evaluations/detection_errors --prefix detection_uncertainty_level_0 --n_episodes 1000 --env_args config_file=experiments/detection_errors/env_config_level_0.yaml --weight_file training/detection_errors/detection_uncertainty_level_0_best.pt --seed 1234 --video
        gnome-terminal -- python3 evaluate.py rl --output_folder evaluations/detection_errors --prefix detection_uncertainty_level_1 --n_episodes 1000 --env_args config_file=experiments/detection_errors/env_config_level_1.yaml --weight_file training/detection_errors/detection_uncertainty_level_1_best.pt --seed 1234 --video
        gnome-terminal -- python3 evaluate.py rl --output_folder evaluations/detection_errors --prefix detection_uncertainty_level_2 --n_episodes 1000 --env_args config_file=experiments/detection_errors/env_config_level_2.yaml --weight_file training/detection_errors/detection_uncertainty_level_2_best.pt --seed 1234 --video
        gnome-terminal -- python3 evaluate.py rl --output_folder evaluations/detection_errors --prefix detection_uncertainty_level_3 --n_episodes 1000 --env_args config_file=experiments/detection_errors/env_config_level_3.yaml --weight_file training/detection_errors/detection_uncertainty_level_3_best.pt --seed 1234 --video
        gnome-terminal -- python3 evaluate.py rl --output_folder evaluations/detection_errors --prefix detection_uncertainty_level_4 --n_episodes 1000 --env_args config_file=experiments/detection_errors/env_config_level_4.yaml --weight_file training/detection_errors/detection_uncertainty_level_4_best.pt --seed 1234 --video
    fi
    if contains_option "fields2cover" "$@"; then
        gnome-terminal -- python3 evaluate.py fields2cover --output_folder evaluations/detection_errors --prefix detection_uncertainty_level_0 --n_episodes 1000 --env_args config_file=experiments/detection_errors/env_config_level_0.yaml --seed 1234 --video
        gnome-terminal -- python3 evaluate.py fields2cover --output_folder evaluations/detection_errors --prefix detection_uncertainty_level_1 --n_episodes 1000 --env_args config_file=experiments/detection_errors/env_config_level_1.yaml --seed 1234 --video
        gnome-terminal -- python3 evaluate.py fields2cover --output_folder evaluations/detection_errors --prefix detection_uncertainty_level_2 --n_episodes 1000 --env_args config_file=experiments/detection_errors/env_config_level_2.yaml --seed 1234 --video
        gnome-terminal -- python3 evaluate.py fields2cover --output_folder evaluations/detection_errors --prefix detection_uncertainty_level_3 --n_episodes 1000 --env_args config_file=experiments/detection_errors/env_config_level_3.yaml --seed 1234 --video
        gnome-terminal -- python3 evaluate.py fields2cover --output_folder evaluations/detection_errors --prefix detection_uncertainty_level_4 --n_episodes 1000 --env_args config_file=experiments/detection_errors/env_config_level_4.yaml --seed 1234 --video
    fi
fi

if contains_option "stopping_criteria" "$@"; then
    if contains_option "rl" "$@"; then
        gnome-terminal -- python3 evaluate.py rl --output_folder evaluations/stopping_criteria --prefix stopping_criteria_all_objects  --n_episodes 1000--env_args config_file=experiments/stopping_criteria/env_config_all_objects.yaml --weight_file training/stopping_criteria/stopping_criteria_all_objects_best.pt --seed 1234 --video
        gnome-terminal -- python3 evaluate.py rl --output_folder evaluations/stopping_criteria --prefix stopping_criteria_coverage_50 --n_episodes 1000 --env_args config_file=experiments/stopping_criteria/env_config_coverage_50.yaml --weight_file training/stopping_criteria/stopping_criteria_coverage_50_best.pt --seed 1234 --video
        gnome-terminal -- python3 evaluate.py rl --output_folder evaluations/stopping_criteria --prefix stopping_criteria_coverage_75 --n_episodes 1000 --env_args config_file=experiments/stopping_criteria/env_config_coverage_75.yaml --weight_file trainingstopping_criteria//stopping_criteria_coverage_75_best.pt --seed 1234 --video
        gnome-terminal -- python3 evaluate.py rl --output_folder evaluations/stopping_criteria --prefix stopping_criteria_land --n_episodes 1000 --env_args config_file=experiments/stopping_criteria/env_config_land.yaml --weight_file training/stopping_criteria/stopping_criteria_land_best.pt --seed 1234 --video
        gnome-terminal -- python3 evaluate.py rl --output_folder evaluations/stopping_criteria --prefix stopping_criteria_new_objects_15 --n_episodes 1000 --env_args config_file=experiments/stopping_criteria/env_config_new_objects_15.yaml --weight_file training/stopping_criteria/stopping_criteria_new_objects_15_best.pt --seed 1234 --video
        gnome-terminal -- python3 evaluate.py rl --output_folder evaluations/stopping_criteria --prefix stopping_criteria_new_objects_25 --n_episodes 1000 --env_args config_file=experiments/stopping_criteria/env_config_new_objects_25.yaml --weight_file training/stopping_criteria/stopping_criteria_new_objects_25_best.pt --seed 1234 --video
        gnome-terminal -- python3 evaluate.py rl --output_folder evaluations/stopping_criteria --prefix stopping_criteria_new_objects_50 --n_episodes 1000 --env_args config_file=experiments/stopping_criteria/env_config_new_objects_50.yaml --weight_file training/stopping_criteria/stopping_criteria_new_objects_50_best.pt --seed 1234 --video
        gnome-terminal -- python3 evaluate.py rl --output_folder evaluations/stopping_criteria --prefix stopping_criteria_new_objects_75 --n_episodes 1000 --env_args config_file=experiments/stopping_criteria/env_config_new_objects_75.yaml --weight_file training/stopping_criteria/stopping_criteria_new_objects_75_best.pt --seed 1234 --video
        gnome-terminal -- python3 evaluate.py rl --output_folder evaluations/stopping_criteria --prefix stopping_criteria_new_objects_150 --n_episodes 1000 --env_args config_file=experiments/stopping_criteria/env_config_new_objects_150.yaml --weight_file training/stopping_criteria/stopping_criteria_new_objects_150_best.pt --seed 1234 --video
    fi
    if contains_option "fields2cover" "$@"; then
        gnome-terminal -- python3 evaluate.py fields2cover --output_folder evaluations/stopping_criteria --prefix stopping_criteria_all_objects --n_episodes 1000 --env_args config_file=experiments/stopping_criteria/env_config_all_objects.yaml --seed 1234 --video
        gnome-terminal -- python3 evaluate.py fields2cover --output_folder evaluations/stopping_criteria --prefix stopping_criteria_coverage_50 --n_episodes 1000 --env_args config_file=experiments/stopping_criteria/env_config_coverage_50.yaml --seed 1234 --video
        gnome-terminal -- python3 evaluate.py fields2cover --output_folder evaluations/stopping_criteria --prefix stopping_criteria_coverage_75 --n_episodes 1000 --env_args config_file=experiments/stopping_criteria/env_config_coverage_75.yaml --seed 1234 --video
        gnome-terminal -- python3 evaluate.py fields2cover --output_folder evaluations/stopping_criteria --prefix stopping_criteria_land --n_episodes 1000 --env_args config_file=experiments/stopping_criteria/env_config_land.yaml --seed 1234 --video
        gnome-terminal -- python3 evaluate.py fields2cover --output_folder evaluations/stopping_criteria --prefix stopping_criteria_new_objects_15 --n_episodes 1000 --env_args config_file=experiments/stopping_criteria/env_config_new_objects_15.yaml --seed 1234 --video
        gnome-terminal -- python3 evaluate.py fields2cover --output_folder evaluations/stopping_criteria --prefix stopping_criteria_new_objects_25 --n_episodes 1000 --env_args config_file=experiments/stopping_criteria/env_config_new_objects_25.yaml --seed 1234 --video
        gnome-terminal -- python3 evaluate.py fields2cover --output_folder evaluations/stopping_criteria --prefix stopping_criteria_new_objects_50 --n_episodes 1000 --env_args config_file=experiments/stopping_criteria/env_config_new_objects_50.yaml --seed 1234 --video
        gnome-terminal -- python3 evaluate.py fields2cover --output_folder evaluations/stopping_criteria --prefix stopping_criteria_new_objects_75 --n_episodes 1000 --env_args config_file=experiments/stopping_criteria/env_config_new_objects_75.yaml --seed 1234 --video
        gnome-terminal -- python3 evaluate.py fields2cover --output_folder evaluations/stopping_criteria --prefix stopping_criteria_new_objects_150 --n_episodes 1000 --env_args config_file=experiments/stopping_criteria/env_config_new_objects_150.yaml --seed 1234 --video
    fi
fi


if contains_option "showcase" "$@"; then
    if [[ -z "${DATA_HOME}" ]]; then
        echo "DATA_HOME is not set. Cannot evaluate."
        exit 1
    fi

    if contains_option "rl" "$@"; then
        python3 evaluate.py rl --output_folder evaluations/showcase --prefix clustered_1 --n_episodes 1 --env_args config_file=experiments/showcase/clustered_1.yaml --weight_file training/stopping_criteria/stopping_criteria_land_best.pt --seed 1234 --video --deterministic --render --save_object_map --save_gt_map
        python3 evaluate.py rl --output_folder evaluations/showcase --prefix clustered_2 --n_episodes 1 --env_args config_file=experiments/showcase/clustered_2.yaml --weight_file training/stopping_criteria/stopping_criteria_land_best.pt --seed 1234 --video --deterministic --render --save_object_map --save_gt_map
        python3 evaluate.py rl --output_folder evaluations/showcase --prefix clustered_3 --n_episodes 1 --env_args config_file=experiments/showcase/clustered_3.yaml --weight_file training/stopping_criteria/stopping_criteria_land_best.pt --seed 1234 --video --deterministic --render --save_object_map --save_gt_map
        python3 evaluate.py rl --output_folder evaluations/showcase --prefix clustered_4 --n_episodes 1 --env_args config_file=experiments/showcase/clustered_4.yaml --weight_file training/stopping_criteria/stopping_criteria_land_best.pt --seed 1234 --video --deterministic --render --save_object_map --save_gt_map
    fi
    if contains_option "fields2cover" "$@"; then
        python3 evaluate.py fields2cover --output_folder evaluations/showcase --prefix clustered_1 --n_episodes 1 --env_args config_file=experiments/showcase/clustered_1.yaml --seed 1234 --video --deterministic --render --save_object_map --save_gt_map
        python3 evaluate.py fields2cover --output_folder evaluations/showcase --prefix clustered_2 --n_episodes 1 --env_args config_file=experiments/showcase/clustered_2.yaml --seed 1234 --video --deterministic --render --save_object_map --save_gt_map
        python3 evaluate.py fields2cover --output_folder evaluations/showcase --prefix clustered_3 --n_episodes 1 --env_args config_file=experiments/showcase/clustered_3.yaml --seed 1234 --video --deterministic --render --save_object_map --save_gt_map
        python3 evaluate.py fields2cover --output_folder evaluations/showcase --prefix clustered_4 --n_episodes 1 --env_args config_file=experiments/showcase/clustered_4.yaml --seed 1234 --video --deterministic --render --save_object_map --save_gt_map
    fi
fi
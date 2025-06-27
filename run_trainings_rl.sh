#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 [distributions] [observation_resolution] [detection_errors] [prior_knowledge_quality] [stopping_criteria]"
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

run_training() {
    export WANDB_NAME="$1"
    export WANDB_TAGS="$2"
    export SDL_VIDEODRIVER="dummy"

    python3 train_rl.py --algo custom_dqn \
        --env DroneGridEnv-v0 \
        --env-kwargs config_file:"'$3'" \
        --track \
        --wandb-project-name paper_3 \
        --gym-packages drone_grid_env \
        --conf-file $4 \
        --eval-freq 0 \
        --vec-env subproc \
        --progress
}

export -f run_training


if contains_option "distributions" "$@"; then
    gnome-terminal -- bash -c 'run_training "distribution_strong" "distribution" "experiments/distributions/env_config_strong.yaml" "experiments/distributions/dqn_strong.yaml"; sleep 20' 
    gnome-terminal -- bash -c 'run_training "distribution_medium" "distribution" "experiments/distributions/env_config_medium.yaml" "experiments/distributions/dqn_medium.yaml"; sleep 20'
    gnome-terminal -- bash -c 'run_training "distribution_random" "distribution" "experiments/distributions/env_config_random.yaml" "experiments/distributions/dqn_random.yaml"; sleep 20'
fi

if contains_option "detection_errors" "$@"; then
    gnome-terminal -- bash -c 'run_training "detection_uncertainty_level_0" "detection_errors" "experiments/detection_errors/env_config_level_0.yaml" "experiments/detection_errors/dqn_level_0.yaml"; sleep 20'
    gnome-terminal -- bash -c 'run_training "detection_uncertainty_level_1" "detection_errors" "experiments/detection_errors/env_config_level_1.yaml" "experiments/detection_errors/dqn_level_1.yaml"; sleep 20'
    gnome-terminal -- bash -c 'run_training "detection_uncertainty_level_2" "detection_errors" "experiments/detection_errors/env_config_level_2.yaml" "experiments/detection_errors/dqn_level_2.yaml"; sleep 20'
    gnome-terminal -- bash -c 'run_training "detection_uncertainty_level_3" "detection_errors" "experiments/detection_errors/env_config_level_3.yaml" "experiments/detection_errors/dqn_level_3.yaml"; sleep 20'
    gnome-terminal -- bash -c 'run_training "detection_uncertainty_level_4" "detection_errors" "experiments/detection_errors/env_config_level_4.yaml" "experiments/detection_errors/dqn_level_4.yaml"; sleep 20'
fi

if contains_option "prior_knowledge_quality" "$@"; then
    gnome-terminal -- bash -c 'run_training "prior_knowledge_uncertainty_level_0" "prior_knowledge_quality" "experiments/prior_knowledge_quality/env_config_level_0.yaml" "experiments/prior_knowledge_quality/dqn_level_0.yaml"; sleep 20'
    gnome-terminal -- bash -c 'run_training "prior_knowledge_uncertainty_level_1" "prior_knowledge_quality" "experiments/prior_knowledge_quality/env_config_level_1.yaml" "experiments/prior_knowledge_quality/dqn_level_1.yaml"; sleep 20'
    gnome-terminal -- bash -c 'run_training "prior_knowledge_uncertainty_level_2" "prior_knowledge_quality" "experiments/prior_knowledge_quality/env_config_level_2.yaml" "experiments/prior_knowledge_quality/dqn_level_2.yaml"; sleep 20'
    gnome-terminal -- bash -c 'run_training "prior_knowledge_uncertainty_level_3" "prior_knowledge_quality" "experiments/prior_knowledge_quality/env_config_level_3.yaml" "experiments/prior_knowledge_quality/dqn_level_3.yaml"; sleep 20'
    gnome-terminal -- bash -c 'run_training "prior_knowledge_uncertainty_level_4" "prior_knowledge_quality" "experiments/prior_knowledge_quality/env_config_level_4.yaml" "experiments/prior_knowledge_quality/dqn_level_4.yaml"; sleep 20'
fi

if contains_option "stopping_criteria" "$@"; then
    gnome-terminal -- bash -c 'run_training "stopping_criteria_all_objects" "stopping_criteria" "experiments/stopping_criteria/env_config_all_objects.yaml" "experiments/stopping_criteria/dqn_all_objects.yaml"; sleep 20'
    gnome-terminal -- bash -c 'run_training "stopping_criteria_coverage_50" "stopping_criteria" "experiments/stopping_criteria/env_config_coverage_50.yaml" "experiments/stopping_criteria/dqn_coverage_50.yaml"; sleep 20'
    gnome-terminal -- bash -c 'run_training "stopping_criteria_coverage_75" "stopping_criteria" "experiments/stopping_criteria/env_config_coverage_75.yaml" "experiments/stopping_criteria/dqn_coverage_75.yaml"; sleep 20'
    gnome-terminal -- bash -c 'run_training "stopping_criteria_land" "stopping_criteria" "experiments/stopping_criteria/env_config_land.yaml" "experiments/stopping_criteria/dqn_land.yaml"; sleep 20'
    gnome-terminal -- bash -c 'run_training "stopping_criteria_new_objects_15" "stopping_criteria" "experiments/stopping_criteria/env_config_new_objects_15.yaml" "experiments/stopping_criteria/dqn_new_objects_15.yaml"; sleep 20'
    gnome-terminal -- bash -c 'run_training "stopping_criteria_new_objects_25" "stopping_criteria" "experiments/stopping_criteria/env_config_new_objects_25.yaml" "experiments/stopping_criteria/dqn_new_objects_25.yaml"; sleep 20'
    gnome-terminal -- bash -c 'run_training "stopping_criteria_new_objects_50" "stopping_criteria" "experiments/stopping_criteria/env_config_new_objects_50.yaml" "experiments/stopping_criteria/dqn_new_objects_50.yaml"; sleep 20'
    gnome-terminal -- bash -c 'run_training "stopping_criteria_new_objects_75" "stopping_criteria" "experiments/stopping_criteria/env_config_new_objects_75.yaml" "experiments/stopping_criteria/dqn_new_objects_75.yaml"; sleep 20'
    gnome-terminal -- bash -c 'run_training "stopping_criteria_new_objects_150" "stopping_criteria" "experiments/stopping_criteria/env_config_new_objects_150.yaml" "experiments/stopping_criteria/dqn_new_objects_150.yaml"; sleep 20'
fi

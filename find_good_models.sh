#!/bin/bash

BALANCE_TYPE="over_smote"

echo $BALANCE_TYPE
python find_good_model.py --model_type knn --rebalance_type $BALANCE_TYPE
python find_good_model.py --model_type svc --rebalance_type $BALANCE_TYPE
python find_good_model.py --model_type lgbm --rebalance_type $BALANCE_TYPE

BALANCE_TYPE="combine_smotenn"

echo $BALANCE_TYPE
python find_good_model.py --model_type knn --rebalance_type $BALANCE_TYPE
python find_good_model.py --model_type svc --rebalance_type $BALANCE_TYPE
python find_good_model.py --model_type lgbm --rebalance_type $BALANCE_TYPE

BALANCE_TYPE="under_random"

echo $BALANCE_TYPE
python find_good_model.py --model_type knn --rebalance_type $BALANCE_TYPE
python find_good_model.py --model_type svc --rebalance_type $BALANCE_TYPE
python find_good_model.py --model_type lgbm --rebalance_type $BALANCE_TYPE

BALANCE_TYPE="over_random"

echo $BALANCE_TYPE
python find_good_model.py --model_type knn --rebalance_type $BALANCE_TYPE
python find_good_model.py --model_type svc --rebalance_type $BALANCE_TYPE
python find_good_model.py --model_type lgbm --rebalance_type $BALANCE_TYPE

BALANCE_TYPE="none"

echo $BALANCE_TYPE
python find_good_model.py --model_type knn --rebalance_type $BALANCE_TYPE
python find_good_model.py --model_type svc --rebalance_type $BALANCE_TYPE
python find_good_model.py --model_type lgbm --rebalance_type $BALANCE_TYPE


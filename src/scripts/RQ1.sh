NUM_TEST=5
#NUM_TEST is the number of test points in RQ1. Use a small value for a quick test, or use 100 for a full test.
#NUM_TEST=100
python -u RQ1.py --model MF --dataset yelp --num_test ${NUM_TEST} --num_steps_train 80000 --num_steps_retrain 24000 > RQ1_MF_yelp.log;
python -u RQ1.py --model MF --dataset movielens --num_test ${NUM_TEST} --num_steps_train 80000 --num_steps_retrain 24000 > RQ1_MF_movielens.log;
python -u RQ1.py --model NCF --dataset yelp --num_test ${NUM_TEST} --num_steps_train 120000 --num_steps_retrain 18000  > RQ1_NCF_yelp.log;
python -u RQ1.py --model NCF --dataset movielens --num_test ${NUM_TEST} --num_steps_train 120000 --num_steps_retrain 18000 --sort_test_case 1 > RQ1_NCF_movielens.log;

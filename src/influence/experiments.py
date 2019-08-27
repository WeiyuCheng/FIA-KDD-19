import numpy as np
import os

def record_time_cost(model, test_idx, iter_to_load, force_refresh=False, random_seed=17):
    np.random.seed(random_seed)
    model.load_checkpoint(iter_to_load)
    y_test = model.data_sets.test.labels[test_idx]
    print('Test label: %s' % y_test)
    approx_params = {'batch_size': model.batch_size, "damping": model.damping}
    model.get_influence_on_test_loss(
        [test_idx],
        np.arange(len(model.data_sets.train.labels)),
        force_refresh=force_refresh,
        approx_params=approx_params)
    return 0

def test_retraining(model, test_idx, iter_to_load, retrain_times, force_refresh=False,
                    num_to_remove=50, num_steps=1000, random_seed=17,
                    remove_type='random', reset_adam=0, load_checkpoint=True):
    np.random.seed(random_seed)

    model.load_checkpoint(iter_to_load)
    sess = model.sess

    y_test = model.data_sets.test.labels[test_idx]
    print('Test label: %s' % y_test)

    ## Or, randomly remove training examples
    if remove_type == 'random':
        indices_to_remove = np.random.choice(model.num_train_examples, size=num_to_remove, replace=False)
        predicted_y_diffs = model.get_influence_on_test_loss(
            [test_idx],
            indices_to_remove,
            force_refresh=force_refresh)
    ## Or, remove the most influential training examples
    elif remove_type == 'maxinf':
        approx_params = {'batch_size': model.batch_size, "damping": model.damping}
        predicted_y_diffs = model.get_influence_on_test_loss(
            [test_idx],
            np.arange(len(model.data_sets.train.labels)),
            force_refresh=force_refresh,
            approx_params=approx_params)
        np.save("output/%s-[%s]-_total_y_diffs" % (model.model_name, test_idx), predicted_y_diffs)
        print("predicted differences are saved for test point %d: %s,%s" % (
            test_idx, model.data_sets.test._x[test_idx], model.data_sets.test._labels[test_idx],))
        indices_to_remove = np.argsort(np.abs(predicted_y_diffs))[-num_to_remove:]
        indices_to_remove = indices_to_remove[::-1]
        predicted_y_diffs = predicted_y_diffs[indices_to_remove]
    else:
        raise (ValueError, 'remove_type not well specified')
    actual_y_diffs = np.zeros([num_to_remove])
    print("Indices to remove are:")
    print(model.train_indices_of_test_case[indices_to_remove])

    # Sanity check
    test_feed_dict = model.fill_feed_dict_with_one_ex(
        model.data_sets.test,
        test_idx)
    test_y_val, params_val = sess.run([model.logits, model.params], feed_dict=test_feed_dict)
    print("Prediction for the test case is:")
    print(test_y_val)
    train_loss_val = sess.run(model.total_loss, feed_dict=model.all_train_feed_dict)

    retrained_test_y_val, retrained_train_loss_val = [], []

    if not load_checkpoint:
        retrained_test_y_val_this = model.sess.run(model.logits, feed_dict=test_feed_dict)
        retrained_train_loss_val_this = model.sess.run(model.total_loss, feed_dict=model.all_train_feed_dict)
        retrained_test_y_val.append(retrained_test_y_val_this)
        retrained_train_loss_val.append(retrained_train_loss_val_this)
    else:
        for i in range(retrain_times):
            if reset_adam:
                model.sess.run(model.reset_optimizer_op)
            checkpoint_to_load = model.checkpoint_file + '_%d_%d_retrain_%d' % (iter_to_load, num_steps, i)
            if os.path.isfile("%s.index" % checkpoint_to_load):
                print('Checkpoint found, loading...')
                model.saver.restore(model.sess, checkpoint_to_load)
            else:
                print('Checkpoint not found, start retraining...')
                model.retrain(num_steps=num_steps, feed_dict=model.all_train_feed_dict)
                model.saver.save(model.sess, model.checkpoint_file + 'retrain_%d' % i)
            retrained_test_y_val_this = model.sess.run(model.logits, feed_dict=test_feed_dict)
            retrained_train_loss_val_this = model.sess.run(model.total_loss, feed_dict=model.all_train_feed_dict)
            retrained_test_y_val.append(retrained_test_y_val_this)
            retrained_train_loss_val.append(retrained_train_loss_val_this)
            model.load_checkpoint(iter_to_load, do_checks=False)
            print(f'model\'s global step is {model.global_step.eval(session=model.sess)}')

    if load_checkpoint:
        bias_retrain = np.array(retrained_test_y_val).mean() - test_y_val
    else:
        bias_retrain = 0
    print('Sanity check: what happens if you train the model a bit more?')
    print('Prediction on test idx with original model    : %s' % test_y_val)
    print('Prediction on test idx with retrained model   : %s' % retrained_test_y_val)
    print(
        'Difference in prediction after retraining     : %s' % bias_retrain)
    print('===')
    print('Total loss on training set with original model    : %s' % train_loss_val)
    print('Total loss on training with retrained model   : %s' % retrained_train_loss_val)
    print(
        'Difference in train loss after retraining     : %s' % (
                np.array(retrained_train_loss_val).mean() - train_loss_val))

    print('These differences should be close to 0.\n')

    # Retraining experiment
    for counter, idx_to_remove in enumerate(indices_to_remove):

        print("=== #%s ===" % counter)
        print('Predicted difference in y (influence): %s' % predicted_y_diffs[counter])
        print(
            'Retraining without train_idx %s (label %s):' % (
                model.train_indices_of_test_case[idx_to_remove],
                model.data_sets.train.labels[model.train_indices_of_test_case[idx_to_remove]]))

        train_feed_dict = model.fill_feed_dict_with_all_but_one_ex(model.data_sets.train,
                                                                   model.train_indices_of_test_case[idx_to_remove])

        retrained_test_y_val, retrained_params_val = [], []
        for i in range(retrain_times):
            if reset_adam:
                print('Resetting adam parameters...')
                model.sess.run(model.reset_optimizer_op)
            model.retrain(num_steps=num_steps, feed_dict=train_feed_dict)
            retrained_test_y_val_this, retrained_params_val_this = model.sess.run([model.logits, model.params],
                                                                                     feed_dict=test_feed_dict)
            retrained_test_y_val.append(retrained_test_y_val_this)
            retrained_params_val.append(retrained_params_val_this)
            if load_checkpoint:
                model.load_checkpoint(iter_to_load, do_checks=False)
                print(f'model\'s global step is {model.global_step.eval(session=model.sess)}')

        # remove few abnormal values
        retrained_test_y_val = np.asarray(retrained_test_y_val)
        retrained_test_y_val = retrained_test_y_val[~np.isnan(retrained_test_y_val)]
        actual_y_diffs[counter] = np.array(retrained_test_y_val).mean() - test_y_val - bias_retrain
        if np.abs(predicted_y_diffs[counter]) > 1:
            predicted_y_diffs[counter] = 0

        print(
            'Diff in params: %s' % np.linalg.norm(
                np.concatenate(params_val) - np.concatenate(retrained_params_val_this)))
        print('y on test idx with original model    : %s' % test_y_val)
        print('y on test idx with retrained model   : %s' % retrained_test_y_val)
        print('Difference in y after retraining     : %s' % actual_y_diffs[counter])
        print('Predicted difference in y (influence): %s' % predicted_y_diffs[counter])

    return actual_y_diffs, predicted_y_diffs, indices_to_remove

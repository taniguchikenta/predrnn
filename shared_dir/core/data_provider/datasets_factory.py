from core.data_provider import kth_action, mnist, bair, products
# import kth_action, mnist, bair, products

datasets_map = {
    'mnist': mnist,
    'action': kth_action,
    'bair': bair,
    'products':products,
}


def data_provider(dataset_name, train_data_paths, valid_data_paths, test_data_paths, test_dir_name, batch_size,
                  img_width, seq_length, injection_action, is_training=True):
    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)
    train_data_list = train_data_paths.split(',')
    valid_data_list = valid_data_paths.split(',')
    test_data_list = test_data_paths.split(',')

    if dataset_name == 'mnist':
        test_input_param = {'paths': valid_data_list,
                            'minibatch_size': batch_size,
                            'input_data_type': 'float32',
                            'is_output_sequence': True,
                            'name': dataset_name + 'test iterator'}
        test_input_handle = datasets_map[dataset_name].InputHandle(test_input_param)
        test_input_handle.begin(do_shuffle=False)
        if is_training:
            train_input_param = {'paths': train_data_list,
                                 'minibatch_size': batch_size,
                                 'input_data_type': 'float32',
                                 'is_output_sequence': True,
                                 'name': dataset_name + ' train iterator'}
            train_input_handle = datasets_map[dataset_name].InputHandle(train_input_param)
            train_input_handle.begin(do_shuffle=True)
            return train_input_handle, test_input_handle
        else:
            return test_input_handle

    if dataset_name == 'action':
        input_param = {'paths': valid_data_list,
                       'image_width': img_width,
                       'minibatch_size': batch_size,
                       'seq_length': seq_length,
                       'input_data_type': 'float32',
                       'name': dataset_name + ' iterator'}
        input_handle = datasets_map[dataset_name].DataProcess(input_param)
        if is_training:
            train_input_handle = input_handle.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return train_input_handle, test_input_handle
        else:
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=True)
            return test_input_handle

    if dataset_name == 'products':
        # test_input_param = {'valid_data_paths': valid_data_list,
        #                     'train_data_paths': train_data_list,
        #                     'test_data_paths': test_data_list,
        #                     'batch_size': batch_size,
        #                     'image_width': img_width,
        #                     'seq_length': seq_length,
        #                     'input_data_type': 'float32',
        #                     'name': dataset_name + 'test iterator'}
        # input_handle_test = datasets_map[dataset_name].DataProcess(test_input_param)
        # test_input_handle = input_handle_test.get_test_input_handle()
        # test_input_handle.begin(do_shuffle=True)######################################33
        if is_training:
            test_input_param = {'valid_data_paths': valid_data_list,
                                'train_data_paths': train_data_list,
                                'test_data_paths': test_data_list,
                                'test_dir_name': test_dir_name,
                                'batch_size': batch_size,
                                'image_width': img_width,
                                'seq_length': seq_length,
                                'input_data_type': 'float32',
                                'name': dataset_name + 'test iterator'}
            input_handle_test = datasets_map[dataset_name].DataProcess(test_input_param)
            test_input_handle = input_handle_test.get_test_input_handle()
            test_input_handle.begin(do_shuffle=True)
            train_input_param = {'valid_data_paths': valid_data_list,
                                'train_data_paths': train_data_list,
                                'test_data_paths': test_data_list,
                                'test_dir_name': test_dir_name,
                                'batch_size': batch_size,
                                'image_width': img_width,
                                'seq_length': seq_length,
                                'input_data_type': 'float32',
                                'name': dataset_name + 'train iterator'}
            input_handle_train = datasets_map[dataset_name].DataProcess(train_input_param)
            train_input_handle = input_handle_train.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            return train_input_handle, test_input_handle
        else:
            test_test_input_param = {'valid_data_paths': valid_data_list,##################################
                                    'train_data_paths': train_data_list,
                                    'test_data_paths': test_data_list,
                                    'test_dir_name': test_dir_name, 
                                    'batch_size': batch_size,
                                    'image_width': img_width,
                                    'seq_length': seq_length,
                                    'input_data_type': 'float32',
                                    'name': dataset_name + 'test iterator'}
            input_handle_test_test = datasets_map[dataset_name].DataProcess(test_test_input_param)
            test_test_input_handle = input_handle_test_test.get_test_test_input_handle()
            test_test_input_handle.begin(do_shuffle=False)
            frames_products_mark_handle = input_handle_test_test.get_frames_products_mark()
            return test_test_input_handle, frames_products_mark_handle
        

    if dataset_name == 'bair':
        test_input_param = {'valid_data_paths': valid_data_list,
                            'train_data_paths': train_data_list,
                            'batch_size': batch_size,
                            'image_width': img_width,
                            'image_height': img_width,
                            'seq_length': seq_length,
                            'injection_action': injection_action,
                            'input_data_type': 'float32',
                            'name': dataset_name + 'test iterator'}
        input_handle_test = datasets_map[dataset_name].DataProcess(test_input_param)
        test_input_handle = input_handle_test.get_test_input_handle()
        test_input_handle.begin(do_shuffle=False)
        if is_training:
            train_input_param = {'valid_data_paths': valid_data_list,
                                 'train_data_paths': train_data_list,
                                 'batch_size': batch_size,
                                 'image_width': img_width,
                                 'image_height': img_width,
                                 'seq_length': seq_length,
                                 'injection_action': injection_action,
                                 'input_data_type': 'float32',
                                 'name': dataset_name + ' train iterator'}
            input_handle_train = datasets_map[dataset_name].DataProcess(train_input_param)
            train_input_handle = input_handle_train.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            return train_input_handle, test_input_handle
        else:
            return test_input_handle